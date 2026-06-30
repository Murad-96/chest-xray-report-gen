# Chest X-Ray Report Generation

Generate a free-text radiology *findings* report from a single chest X-ray, using a
DenseNet-121 image encoder feeding a Transformer decoder, trained end-to-end on the
IU X-Ray dataset.

> **Live demo:** `<streamlit-app-url>`  ·  **Weights:** `hf://<your-username>/cxr-report-gen/best_serving.pt`

This project frames report generation as image captioning over a medical domain: the
model "reads" a radiograph and drafts the kind of findings paragraph a radiologist
would write. It is a learning/portfolio project and **not** a clinical tool.

---

## Results

Test set (1,180 images, greedy decoding):

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|:------:|:------:|:------:|:------:|:------:|:-------:|:-----:|
| 0.342  | 0.206  | 0.135  | 0.092  | 0.177  | 0.290   | 0.239 |

BLEU-4 of 0.092 sits within the seed-to-seed range reported for this architecture on
this dataset (0.073–0.096), reproducing the reference result on a single seed. The
decisive factor in reaching it was training the encoder end-to-end rather than freezing
it (see below).

---

## Architecture

The model is a standard encoder–decoder captioner with three stages: extract visual
features with a CNN, reshape them into a short sequence of "visual tokens," and let a
Transformer decoder attend to those tokens while it writes the report.

```
            Chest X-ray (single view, 224×224, ImageNet-normalized)
                                  │
                                  ▼
        DenseNet-121  (ImageNet-pretrained, FINE-TUNED end-to-end)
                                  │   spatial feature map  (B, 1024, 7, 7)
                                  ▼
        Flatten 7×7 → 49 positions, linear project 1024 → 512
                                  │   49 visual tokens  (B, 49, 512)
                                  │   — the image as a 49-"word" sentence
                                  ▼
        Transformer decoder  (6 layers, d_model=512, 8 heads, d_ff=2048)
          • token embeddings + sinusoidal positional encoding
          • causal self-attention over generated tokens
          • cross-attention into the 49 visual tokens
                                  │
                                  ▼
                    Free-text findings report (autoregressive)
```

**1. ImageNet features.** A DenseNet-121 backbone, pretrained on ImageNet, encodes each
224×224 image into a `(1024, 7, 7)` spatial feature map.

**2. Features → tokens.** The 7×7 grid is flattened into 49 spatial positions and each
position's 1024-dim feature vector is linearly projected to the decoder width (512).
The image therefore becomes a sequence of 49 "visual tokens" — analogous to a 49-word
sentence the decoder can attend over. The visual tokens carry no positional embedding;
the decoder recovers spatial relationships through cross-attention alone.

**3. Transformer decoder.** A 6-layer Transformer decoder generates the report one token
at a time. Each layer does causal self-attention over the tokens generated so far and
cross-attention into the 49 visual tokens (which serve as the key/value memory).
Training uses teacher forcing with cross-entropy (padding ignored); inference decodes
greedily up to 60 tokens, stopping at `<EOS>`.

### End-to-end training is the key design choice

The encoder is **not frozen** — it is fine-tuned jointly with the decoder. This is the
single most important decision in the project:

- **Frozen encoder:** validation BLEU-4 collapses to ~`4e-6`. The model emits a generic
  "everything is normal" template regardless of the image, because off-the-shelf
  ImageNet features don't separate normal from abnormal chest anatomy. Training loss
  alone gives no warning — it decreases normally while the outputs are useless.
- **Encoder fine-tuned end-to-end:** test BLEU-4 rises to ~0.10, reports lengthen from
  ~10–15 to ~30–50 tokens, and the model begins naming specific findings.

The lesson generalizes: for domain-shifted images (medical scans vs. natural photos),
a pretrained backbone is a *starting point*, not a fixed feature extractor, and
loss curves are not a sufficient diagnostic for sequence-generation models —
per-epoch sample inspection caught failures the loss never surfaced.

---

## Data

[IU X-Ray](https://openi.nlm.nih.gov/) chest radiographs paired with their reports,
using the R2Gen-preprocessed split (canonical across recent report-generation papers):

| Split | Studies | Images |
|-------|:-------:|:------:|
| Train | 2,069   | 4,138  |
| Val   | 296     | 592    |
| Test  | 590     | 1,180  |

- **Single-view captioning.** Most studies have two views (frontal + lateral) that share
  one report; each image is captioned independently.
- **Tokenization.** Reports are lowercased and whitespace-normalized; periods become a
  `<SEP>` token; reports are wrapped with `<BOS>`/`<EOS>` and truncated to 60 tokens.
- **Vocabulary.** Built from the training split only, words with frequency < 3 mapped to
  `<UNK>`, giving 975 tokens (including the 5 special tokens).

The dataset is downloaded separately (it is not in this repo). See
[Setup](#setup) for the download step.

---

## Repository layout

```
.
├── configs/
│   ├── transformer.yaml          # full training config (encoder unfrozen)
│   └── transformer_smoke.yaml    # 2-epoch smoke test, inherits the above
├── src/
│   ├── data/iu_xray.py           # dataset, vocabulary, dataloaders
│   ├── models/
│   │   ├── __init__.py           # get_model() factory
│   │   └── transformer.py        # DenseNet-121 encoder + Transformer decoder
│   ├── training/
│   │   ├── train.py              # training entry point
│   │   ├── trainer.py            # train/val loop, early stopping, checkpointing
│   │   ├── config.py             # YAML loading + schema validation
│   │   ├── checkpoint.py         # atomic save / resume
│   │   ├── scheduler.py          # AdamW + warmup-cosine
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── evaluate.py           # test-set metrics + samples.csv
│   │   └── linguistic_metrics.py # BLEU / METEOR / ROUGE-L / CIDEr
│   └── inference.py              # shared load + preprocess + generate core
├── app/
│   ├── streamlit_app.py          # demo UI
│   └── images/                   # example X-rays + findings.json (for the demo)
├── scripts/
│   └── export_for_serving.py     # slim a checkpoint and push to the HF Hub
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
# METEOR and CIDEr need Java on PATH (for pycocoevalcap's tokenizer):
#   e.g.  sudo apt-get install -y default-jre
```

Download the R2Gen IU X-Ray bundle (~1.1 GB) and extract it so you have
`.../iu_xray/annotation.json` and `.../iu_xray/images/`:

```bash
pip install gdown
gdown 1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg     # R2Gen bundle
unzip -q iu_xray.zip
```

Point the code at the dataset via either the config (`data.root`) or an env var:

```bash
export IU_XRAY_ROOT=/path/to/iu_xray
python -m src.data.iu_xray     # self-check: prints vocab size (975) and batch shapes
```

---

## Training

```bash
# fast end-to-end pipeline check (2 epochs, small batch):
python -m src.training.train --config configs/transformer_smoke.yaml

# full run:
python -m src.training.train --config configs/transformer.yaml
```

Setup: AdamW (lr 1e-4, weight decay 0.01), 2-epoch linear warmup then cosine decay to
0.1× base lr, gradient clipping at 1.0, mixed-precision (fp16), batch size 16, up to 30
epochs with val-loss early stopping (patience 10), seed 42. Checkpoints are written to
`outputs/<experiment.name>/` as `last.pt` (every epoch) and `best.pt` (best val loss);
training is resumable with `--resume`, which restores optimizer, scheduler, and RNG
state — convenient for spot/Colab sessions that can disconnect.

## Evaluation

```bash
python -m src.evaluation.evaluate --checkpoint outputs/transformer/best.pt --split test --no-clinical
```

Writes `metrics.json`, `samples.csv` (every image_id / reference / hypothesis), and a
human-readable `summary.txt` alongside the checkpoint.

---

## Demo app

A Streamlit app generates reports live, either for curated example images (shown next to
their ground-truth reports) or for an image you upload.

```bash
streamlit run app/streamlit_app.py
```

The app and the evaluation script share a single inference core
(`src/inference.py::ReportGenerator`), so image preprocessing and decoding are defined in
exactly one place and cannot drift between training, evaluation, and serving.

**Deployment (Streamlit Community Cloud).** The dataset and the full training checkpoint
are too large to commit. Instead, a slimmed serving checkpoint (weights + vocab + config
only, ~130 MB vs. ~500 MB) is hosted on the Hugging Face Hub:

```bash
python scripts/export_for_serving.py --checkpoint outputs/transformer/best.pt \
    --verify --push-to-hub <your-username>/cxr-report-gen
```

Then set the checkpoint location in the app's Streamlit **Secrets**:

```toml
checkpoint = "hf://<your-username>/cxr-report-gen/best_serving.pt"
```

`ReportGenerator.from_checkpoint` detects the `hf://` prefix and downloads (and caches)
the weights at startup — no dataset required to serve, since the vocabulary is rebuilt
from the checkpoint.

---

## Limitations

- **Single-view.** Each image is captioned independently, so the two views of one study
  can yield different — sometimes inconsistent — reports.
- **Small, skewed data.** 4,138 training images from one US academic center, skewed
  toward normal studies; rare findings are underrepresented and the model can default to
  plausible normal-sounding text.
- **No out-of-distribution guard.** The model captions whatever it is given. A non-CXR
  image, or a radiograph rendered very differently from IU X-Ray, will still produce a
  confident (and likely wrong) report.
- **Single seed; greedy decoding.** No multi-seed averaging or beam search.
- **Not for clinical use.** Research/portfolio demonstration only.

---

## Acknowledgments

This work began as a Georgia Tech CS 7643 (Deep Learning) group project comparing three
report-generation decoders (a hierarchical LSTM, this vanilla Transformer, and a
clinical-term-guided Transformer). My contributions to the original project were the data
preprocessing pipeline (tokenization, vocabulary, R2Gen split integration) and the
vanilla Transformer decoder. This repository extracts those components and extends them
into a standalone, trainable, and deployable single-model system, reusing the team's
shared training harness.

- Original group project: <https://github.com/vermouthwang/dl-cxr-report-gen>

## References

- Chen et al., *Generating Radiology Reports via Memory-driven Transformer* (R2Gen), EMNLP 2020 — dataset split.
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — Transformer.
- Huang et al., *Densely Connected Convolutional Networks* (DenseNet), CVPR 2017 — encoder.
- Rajpurkar et al., *CheXNet*, 2017 — DenseNet-121 for chest X-rays.
