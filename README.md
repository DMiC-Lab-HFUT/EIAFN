# EIAFN in Plain PyTorch

This directory is an independent PyTorch implementation of the Explicit-Implicit Associative Fusion Network (EIAFN). It does not depend on PyTorch Lightning and leaves `eiafn-main` unchanged.

## Run

Install dependencies and set every dataset path in the selected YAML file. The `pretrained_model` value may be a Hugging Face model identifier or a local CLIP checkpoint directory.

```powershell
python main.py --config config/wikimel.yaml --output-dir runs
```

Test a saved checkpoint:

```powershell
python main.py --config config/wikimel.yaml --test-only --checkpoint runs/WikiMEL/best.pt
```

## Paper alignment

The implementation follows the manuscript's three implicit scores, three projected-cosine explicit scores, multiplicative explicit-implicit fusion, softmax modality gating, and the trainable branch-weighted contrastive loss. The implicit text and visual branch terms are summed as written in the paper, rather than averaged.
