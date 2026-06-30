"""
Slim a training checkpoint down to a serving-only checkpoint.

A training best.pt carries optimizer/scheduler/scaler/RNG state for resuming —
none of which inference needs. This strips those, keeping only what
src.inference.ReportGenerator.from_checkpoint reads: the model weights, the
vocab state, and the config (plus a little provenance). Typically cuts size by
~3-4x (e.g. ~500 MB -> ~130 MB), which matters for Hub download time and the
~1 GB memory ceiling on Streamlit Community Cloud.

Usage:
    # just slim it (writes <ckpt_dir>/best_serving.pt):
    python scripts/export_for_serving.py --checkpoint outputs/transformer/best.pt

    # slim + sanity-check that it reloads through ReportGenerator:
    python scripts/export_for_serving.py --checkpoint outputs/transformer/best.pt --verify

    # slim + upload straight to a Hugging Face model repo (must be logged in,
    # or have HF_TOKEN set):
    python scripts/export_for_serving.py --checkpoint outputs/transformer/best.pt \
        --push-to-hub <owner>/<repo>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Repo root on path so --verify can import src.inference.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Keys the serving path reads (config, vocab, weights) plus tiny provenance.
# Everything else in a training checkpoint is dropped.
_KEEP_KEYS = [
    "model_state_dict", "vocab_state", "config",
    "epoch", "best_metric", "torch_version", "git_commit", "_schema_version",
]
# Dropped explicitly (these are the heavy resume-only tensors):
#   optimizer_state_dict, scheduler_state_dict, scaler_state_dict, rng_states


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.expanduser().resolve()

    repo_path = (_REPO_ROOT / path).resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return repo_path


def export(checkpoint: str, output: str) -> tuple[Path, Path]:
    ckpt_path = resolve_path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for required in ("model_state_dict", "vocab_state", "config"):
        if required not in state:
            raise KeyError(
                f"Checkpoint missing required key {required!r}; cannot export for serving."
            )

    slim = {k: state[k] for k in _KEEP_KEYS if k in state}
    slim["_serving_export"] = True

    out_path = resolve_path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_path)
    return ckpt_path, out_path


def verify(output: Path) -> None:
    from src.inference import ReportGenerator
    gen = ReportGenerator.from_checkpoint(output, device=torch.device("cpu"))
    print(f"  verify: reloaded OK — model={gen.model_name}, "
          f"{gen.num_params / 1e6:.1f}M params, max_length={gen.max_length}")


def push_to_hub(output: Path, repo_id: str, hub_filename: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(output),
        path_in_repo=hub_filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  pushed to https://huggingface.co/{repo_id} (as {hub_filename})")
    print(f"  -> app checkpoint spec:  hf://{repo_id}/{hub_filename}")


def main():
    p = argparse.ArgumentParser(description="Export a serving-only checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to a training best.pt")
    p.add_argument("--output", default=None,
                   help="Output path (default: <checkpoint_dir>/best_serving.pt)")
    p.add_argument("--verify", action="store_true",
                   help="Reload the slim checkpoint through ReportGenerator as a check")
    p.add_argument("--push-to-hub", default=None, metavar="OWNER/REPO",
                   help="Upload the slim checkpoint to this HF model repo")
    p.add_argument("--hub-filename", default="best_serving.pt",
                   help="Filename to use inside the HF repo (default: best_serving.pt)")
    args = p.parse_args()

    output = args.output or str(Path(args.checkpoint).parent / "best_serving.pt")

    src_path, out_path = export(args.checkpoint, output)
    print("Exported serving checkpoint:")
    print(f"  in : {src_path}  ({size_mb(src_path):.0f} MB)")
    print(f"  out: {out_path}  ({size_mb(out_path):.0f} MB)")

    if args.verify:
        verify(out_path)

    if args.push_to_hub:
        push_to_hub(out_path, args.push_to_hub, args.hub_filename)


if __name__ == "__main__":
    main()