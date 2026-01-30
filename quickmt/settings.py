"""Centralized configuration management using pydantic-settings.

This module provides a type-safe, centralized way to manage all configuration
settings for the quickmt library. Settings can be configured via:
- Environment variables (e.g., MAX_LOADED_MODELS=10)
- .env file in the project root
- Runtime modification of the global settings object

All environment variables are case-insensitive.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    All settings can be overridden via environment variables.
    For example, to set max_loaded_models, use MAX_LOADED_MODELS=10
    """

    # Model Manager Settings
    max_loaded_models: int = 5
    """Maximum number of translation models to keep loaded in memory"""

    device: str = "cpu"
    """Device to use for inference: 'cpu', 'cuda', or 'auto'"""

    compute_type: str = "default"
    """CTranslate2 compute type: 'default', 'int8', 'int8_float16', 'int16', 'float16', 'float32'"""

    inter_threads: int = 1
    """Number of threads to use for inter-op parallelism (simultaneous translations)"""

    intra_threads: int = 4
    """Number of threads to use for intra-op parallelism (within each translation)"""

    # Batch Processing Settings
    max_batch_size: int = 32
    """Maximum batch size for translation requests"""

    batch_timeout_ms: int = 5
    """Timeout in milliseconds to wait for batching additional requests"""

    # Language Identification Settings
    langid_model_path: Optional[str] = None
    """Path to FastText language identification model. If None, uses default cache location"""

    langid_workers: int = 2
    """Number of worker processes for language identification"""

    # Translation Cache Settings
    translation_cache_size: int = 10000
    """Maximum number of translations to cache (LRU eviction)"""

    port: int = 8000
    """Number of threads to use for inter-op parallelism (simultaneous translations)"""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings instance
# This can be imported and used throughout the application
# Settings can be modified at runtime: settings.max_loaded_models = 10
settings = Settings()
