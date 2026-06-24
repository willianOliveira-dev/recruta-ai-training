import argparse
import logging

from src.config.settings import AppSettings
from src.training.pipeline import TrainingPipeline

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recruta AI - Training")
    parser.add_argument("--config", type=str, default="configs/lite.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppSettings.from_yaml(profile_path=args.config)
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
