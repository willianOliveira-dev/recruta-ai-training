import argparse
import logging
import sys

from src.config.settings import AppSettings
from src.training.publisher import ModelPublisher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recruta AI - Publish Model")
    parser.add_argument("--config", type=str, default="configs/lite.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading config from: %s", args.config)
    config = AppSettings.from_yaml(profile_path=args.config)

    try:
        publisher = ModelPublisher(config)
        publisher.publish()
    except Exception as exc:
        logger.error("Publishing failed: %s", str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
