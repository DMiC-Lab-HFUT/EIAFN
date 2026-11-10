import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eiafn.data import EIAFNData
from eiafn.model import EIAFN
from eiafn.trainer import EIAFNTrainer, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test EIAFN with plain PyTorch.")
    parser.add_argument("--config", default="config/richpediamel.yaml")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint used with --test-only.")
    return parser.parse_args()


def main():
    arguments = parse_args()
    config = OmegaConf.load(arguments.config)
    seed_everything(config.seed)
    torch.set_num_threads(1)
    data = EIAFNData(config)
    model = EIAFN(config)
    run_dir = Path(arguments.output_dir) / config.run_name
    trainer = EIAFNTrainer(model, data, config, run_dir)
    if arguments.test_only:
        if not arguments.checkpoint:
            raise ValueError("--checkpoint is required with --test-only")
        state = torch.load(arguments.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(state["model_state"])
    else:
        trainer.fit()
    print(trainer.test())


if __name__ == "__main__":
    main()
