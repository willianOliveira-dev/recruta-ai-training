from pathlib import Path


class ConfigurationLoadError(Exception):
    def __init__(self, config_path: Path, reason: str) -> None:
        self.config_path = config_path
        self.reason = reason
        super().__init__(f"Failed to load config '{config_path}': {reason}")


class IngestionError(Exception):
    def __init__(self, source_name: str, reason: str) -> None:
        self.source_name = source_name
        self.reason = reason
        super().__init__(f"Failed to ingest '{source_name}': {reason}")


class TrainingError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Training failed: {reason}")


class StrategyNotFoundError(Exception):
    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        super().__init__(f"No strategy registered for type '{strategy_name}'")
