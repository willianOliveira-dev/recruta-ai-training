from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

from src.config.settings import DataSourceConfig
from src.shared.errors import IngestionError, StrategyNotFoundError
from src.shared.protocols import DataSourceStrategy

logger = logging.getLogger(__name__)


class HuggingFaceStrategy:
    def ingest(self, source: DataSourceConfig, output_dir: Path) -> Dataset:
        logger.info("Ingesting HF dataset: %s", source.uri)
        return load_dataset(source.uri)


class DatasetIngestor:
    def __init__(self, output_dir: Path = Path("./data/raw")) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._strategies: dict[str, DataSourceStrategy] = {
            "huggingface": HuggingFaceStrategy(),
        }

    def register_strategy(self, name: str, strategy: DataSourceStrategy) -> None:
        self._strategies[name] = strategy

    def ingest_source(self, source: DataSourceConfig) -> Dataset | Path:
        strategy = self._strategies.get(source.source_type)
        if strategy is None:
            raise StrategyNotFoundError(source.source_type)
        try:
            return strategy.ingest(source, self._output_dir)
        except StrategyNotFoundError:
            raise
        except Exception as exc:
            raise IngestionError(source.name, str(exc)) from exc

    def process_all(self, sources: list[DataSourceConfig]) -> dict[str, Dataset | Path]:
        results: dict[str, Dataset | Path] = {}
        for source in sources:
            try:
                results[source.name] = self.ingest_source(source)
            except (IngestionError, StrategyNotFoundError):
                logger.exception("Failed to ingest source: %s", source.name)
        return results
