import argparse
import logging

from src.config.settings import AppSettings
from src.data.ingestion import DatasetIngestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recruta AI - Data Preparation")
    parser.add_argument("--config", type=str, default="configs/lite.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading config from: %s", args.config)
    config = AppSettings.from_yaml(profile_path=args.config)

    ingestor = DatasetIngestor()
    logger.info("Starting data ingestion pipeline")
    ingestor.process_all(sources=config.data.sources)
    logger.info("Data ingestion pipeline completed")


if __name__ == "__main__":
    main()
