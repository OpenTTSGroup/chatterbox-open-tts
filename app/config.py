from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (CHATTERBOX_* prefix) ----------------------------------------
    chatterbox_variant: Literal["standard", "turbo", "multilingual"] = Field(
        default="multilingual",
        description="Selects ChatterboxTTS / ChatterboxTurboTTS / ChatterboxMultilingualTTS.",
    )
    chatterbox_model: str = Field(
        default="",
        description=(
            "HuggingFace repo id or local checkpoint directory. Empty ⇒ "
            "use the variant's default repo (see app/engine.py)."
        ),
    )
    chatterbox_device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    chatterbox_cuda_index: int = Field(default=0, ge=0)
    chatterbox_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    chatterbox_default_language: str = Field(
        default="en",
        description="Fallback language_id for the multilingual variant when the request omits it.",
    )
    chatterbox_prompt_cache_size: int = Field(
        default=16,
        ge=1,
        description="LRU size for per-reference prepare_conditionals() cache.",
    )

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def resolved_device(self) -> str:
        if self.chatterbox_device == "cpu":
            return "cpu"
        if self.chatterbox_device == "mps":
            return "mps"
        if self.chatterbox_device == "cuda":
            return f"cuda:{self.chatterbox_cuda_index}"
        # auto
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.chatterbox_cuda_index}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
