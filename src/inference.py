"""
Shared inference core for chest X-ray report generation.

Loads a trained checkpoint (model weights + vocab + config, all baked in by
src/training/checkpoint.py) and turns a single chest X-ray image into a
generated report string. The Streamlit app (app/streamlit_app.py) and any
service layer (e.g. serve/api.py) both import ReportGenerator from here, so the
preprocessing and decoding logic lives in exactly one place and can't drift
between surfaces.

Usage:
    from src.inference import ReportGenerator
    gen = ReportGenerator.from_checkpoint("outputs/transformer/best.pt")
    report = gen.predict(pil_image_or_path)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from src.data.iu_xray import Vocabulary
from src.models import get_model

# Must match the eval transform used in src/data/iu_xray.py at train time.
# A mismatch here (e.g. different resize or normalization) silently degrades
# every prediction, so these are duplicated deliberately rather than imported,
# and should be kept in sync if the data pipeline's transform ever changes.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

ImageLike = Union[str, Path, Image.Image]


def _resolve_checkpoint_path(spec: str) -> str:
    """
    Resolve a checkpoint location to a local file path.

    Local paths are returned unchanged. Hugging Face Hub references of the form
    `hf://<owner>/<repo>/<filename>` are downloaded and cached via
    huggingface_hub.hf_hub_download, returning the local cache path. This lets
    the same code serve a local checkpoint in development and a Hub-hosted one
    in deployment (e.g. Streamlit Community Cloud) with no code change. For a
    private Hub repo, set the HF_TOKEN env var (or log in) before loading.
    """
    if not spec.startswith("hf://"):
        return spec
    parts = spec[len("hf://"):].split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Malformed hf:// checkpoint spec: {spec!r}. "
            f"Expected hf://<owner>/<repo>/<filename>."
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to load hf:// checkpoints. "
            "Install it with `pip install huggingface_hub`."
        ) from e
    return hf_hub_download(repo_id=repo_id, filename=filename)


class ReportGenerator:
    """Stateful wrapper around a trained model: load once, predict many times."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        vocab: Vocabulary,
        device: torch.device,
        max_length: int,
        model_name: str,
        trained_epoch: Optional[int],
        best_metric: Optional[float],
        num_params: int,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        self.model_name = model_name
        self.trained_epoch = trained_epoch
        self.best_metric = best_metric
        self.num_params = num_params

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        max_length: Optional[int] = None,
    ) -> "ReportGenerator":
        """
        Rebuild model + vocab from a training checkpoint.

        Mirrors the load path in src/evaluation/evaluate.py: the checkpoint
        carries the full config and the vocab state, so nothing external is
        needed to reconstruct the model.
        """
        ckpt_path = Path(_resolve_checkpoint_path(str(checkpoint_path)))
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Train a model first "
                f"(python -m src.training.train --config configs/transformer.yaml), "
                f"or point at an existing best.pt."
            )

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg = state["config"]
        vocab = Vocabulary.from_state_dict(state["vocab_state"])

        model = get_model(
            name=cfg["model"]["name"],
            vocab_size=vocab.size,
            config=cfg["model"].get("config", {}),
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()

        if max_length is None:
            max_length = int(cfg["validation"]["generation_max_length"])

        num_params = sum(p.numel() for p in model.parameters())

        return cls(
            model=model,
            vocab=vocab,
            device=device,
            max_length=max_length,
            model_name=cfg["model"]["name"],
            trained_epoch=state.get("epoch"),
            best_metric=state.get("best_metric"),
            num_params=num_params,
        )

    @torch.no_grad()
    def predict(
        self,
        image: ImageLike,
        max_length: Optional[int] = None,
        beam_size: int = 1,
    ) -> str:
        """
        Generate a report for a single chest X-ray.

        `image` may be a file path, a PIL Image, or any file-like object that
        PIL can open (e.g. a Streamlit upload). Returns the decoded report
        with special tokens stripped.
        """
        tensor = self._prepare(image)
        ml = max_length or self.max_length
        gen_ids_batch = self.model.generate(tensor, max_length=ml, beam_size=beam_size)
        return self.vocab.decode(gen_ids_batch[0])

    def _prepare(self, image: ImageLike) -> torch.Tensor:
        """Open (if needed), convert to RGB, apply the eval transform, add batch dim."""
        if isinstance(image, Image.Image):
            img = image
        else:
            # Image.open accepts a path (str/Path) or a file-like/BytesIO object.
            img = Image.open(image)
        img = img.convert("RGB")
        return _EVAL_TRANSFORM(img).unsqueeze(0).to(self.device)