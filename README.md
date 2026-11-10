## Project structure

```text
eiafn-torch/
├── config/       # Dataset and training configurations
├── eiafn/        # Data processing, model, and training modules
├── main.py       # Training and evaluation entry point
└── requirements.txt
```

## Requirements

Install the required packages:

```powershell
pip install -r requirements.txt
```

Download the CLIP ViT-B/32 checkpoint and configure `pretrained_model` plus all dataset paths in the selected YAML file under `config/`.

## Training

```powershell
python main.py --config config/wikimel.yaml --output-dir runs
```

Available configurations are `wikimel.yaml`, `wikidiverse.yaml`, and `richpediamel.yaml`.

## Evaluation

```powershell
python main.py --config config/wikimel.yaml --test-only --checkpoint runs/WikiMEL/best.pt
```
