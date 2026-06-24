from __future__ import annotations

import logging
from pathlib import Path

import wandb
from datasets import Dataset

from src.config.settings import AppSettings
from src.data.ingestion import DatasetIngestor
from src.data.preparation import DatasetPreparator
from src.models.unsloth_loader import UnslothModelLoader
from src.shared.errors import TrainingError
from src.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: AppSettings) -> None:
        self._config = config

    def _initialize_wandb(self) -> None:
        if self._config.wandb.enabled:
            wandb.init(
                project=self._config.wandb.project,
                name=self._config.wandb.name,
            )

    def _finalize_wandb(self) -> None:
        if self._config.wandb.enabled:
            wandb.finish()

    def _load_model(self) -> tuple[object, object]:
        logger.info("Step 1: Loading base model and applying PEFT")
        loader = UnslothModelLoader(self._config)
        model, tokenizer = loader.load_base_model()
        model = loader.setup_peft(model)
        return model, tokenizer

    def _ingest_datasets(self) -> dict[str, Dataset | Path]:
        logger.info("Step 2: Ingesting datasets")
        ingestor = DatasetIngestor()
        return ingestor.process_all(sources=self._config.data.sources)

    def _prepare_train_dataset(
        self,
        results: dict[str, Dataset | Path],
        tokenizer: object,
    ) -> Dataset:
        first_source = self._config.data.sources[0] if self._config.data.sources else None
        if first_source is None:
            raise TrainingError("No data sources configured")

        raw_dataset = results.get(first_source.name)
        if raw_dataset is None or not isinstance(raw_dataset, Dataset) and not hasattr(raw_dataset, "__getitem__"):
            raise TrainingError(f"Dataset '{first_source.name}' not available in memory")

        logger.info("Step 3: Preparing training data")
        preparator = DatasetPreparator(tokenizer)

        if hasattr(raw_dataset, "keys") and "train" in raw_dataset:
            return preparator.prepare_dataset(raw_dataset["train"])

        return preparator.prepare_dataset(raw_dataset)

    def run(self) -> None:
        self._initialize_wandb()
        try:
            model, tokenizer = self._load_model()
            results = self._ingest_datasets()
            train_dataset = self._prepare_train_dataset(results, tokenizer)

            logger.info("Step 4: Starting training")
            trainer = ModelTrainer(self._config, model, tokenizer, train_dataset)
            trainer.train()
            trainer.save(Path(self._config.output_dir))

            logger.info("Pipeline completed successfully")
        finally:
            self._finalize_wandb()
