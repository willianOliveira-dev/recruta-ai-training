import logging
import os
from pathlib import Path

from src.config.settings import AppSettings
from src.models.unsloth_loader import UnslothModelLoader
from src.shared.errors import ConfigurationLoadError

logger = logging.getLogger(__name__)


class ModelPublisher:
    def __init__(self, config: AppSettings) -> None:
        self._config = config
        self._verify_config()

    def _verify_config(self) -> None:
        if not self._config.publish.enabled:
            raise ConfigurationLoadError(
                Path("configs"), "Publishing is not enabled in the configuration."
            )
        
        token = os.getenv(self._config.publish.token_env_var)
        if not token:
            raise ConfigurationLoadError(
                Path("configs"),
                f"Hugging Face token not found in environment variable: {self._config.publish.token_env_var}",
            )

    def publish(self) -> None:
        logger.info("Starting model publishing to Hugging Face")
        logger.info("Target Repository: %s", self._config.publish.repo_id)

        # Usamos o Loader para carregar o modelo salvo (base + lora local)
        # Como o modelo treinado já está salvo localmente na pasta `outputs`, 
        # para a extração correta com o Unsloth precisamos usar `load_base_model()` modificado 
        # ou a própria função de unsloth `FastLanguageModel.from_pretrained` apontando para o output_dir.
        
        from unsloth import FastLanguageModel

        logger.info("Loading trained model from: %s", self._config.output_dir)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._config.output_dir,
            max_seq_length=self._config.training.max_seq_length,
            dtype=None,
            load_in_4bit=self._config.model.load_in_4bit,
        )

        token = os.getenv(self._config.publish.token_env_var)

        if self._config.publish.merge_before_push:
            logger.info("Merging LoRA adapters into 16-bit base model before pushing...")
            model.push_to_hub_merged(
                self._config.publish.repo_id,
                tokenizer,
                save_method="merged_16bit",
                token=token,
            )
        else:
            logger.info("Pushing LoRA adapters only...")
            model.push_to_hub(self._config.publish.repo_id, token=token)
            tokenizer.push_to_hub(self._config.publish.repo_id, token=token)

        logger.info("Model successfully published to Hugging Face Hub!")
