from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from app.config import Settings


log = logging.getLogger(__name__)

# Chatterbox's S3GEN output is fixed at 24 kHz for all variants.
_SAMPLE_RATE = 24000

# generate() kwargs the service is willing to forward to the model, per variant.
# Anything outside this whitelist is silently dropped — keeps the upstream
# model from logging ignored-parameter warnings and lets the SpeechRequest
# schema stay uniform across variants.
_APPLICABLE_KWARGS: dict[str, frozenset[str]] = {
    "standard": frozenset(
        {"repetition_penalty", "min_p", "top_p", "exaggeration", "cfg_weight", "temperature"}
    ),
    "turbo": frozenset(
        {"repetition_penalty", "top_p", "temperature", "top_k", "norm_loudness"}
    ),
    "multilingual": frozenset(
        {"exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p"}
    ),
}


def _default_exaggeration(variant: str) -> float:
    # Matches the generate() signatures in engine/src/chatterbox/*.py.
    return 0.0 if variant == "turbo" else 0.5


# Default HuggingFace repo id per variant. Mirrors the REPO_ID constants
# that upstream Chatterbox bakes into each class's from_pretrained().
_DEFAULT_REPO_PER_VARIANT: dict[str, str] = {
    "standard": "ResembleAI/chatterbox",
    "turbo": "ResembleAI/chatterbox-turbo",
    "multilingual": "ResembleAI/chatterbox",
}

# Files that each variant's ``from_local()`` reads. Used as allow_patterns
# for snapshot_download() to avoid pulling unrelated weights from the repo.
_HF_ALLOW_PATTERNS: dict[str, list[str]] = {
    "standard": [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ],
    "turbo": ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    "multilingual": [
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ],
}


class ChatterboxEngine:
    """Thin async wrapper around Chatterbox standard / turbo / multilingual."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._variant = settings.chatterbox_variant
        self._device_str = settings.resolved_device
        self._dtype_str = settings.chatterbox_dtype
        self._default_language = settings.chatterbox_default_language
        self._inference_lock = threading.Lock()

        self._model = self._load_model()

        if self._model.conds is None:
            raise RuntimeError(
                "loaded Chatterbox checkpoint did not contain conds.pt; "
                "the built-in 'default' voice cannot be served"
            )
        # Snapshot the shipped default speaker so it can be restored after any
        # file-voice call overwrites self._model.conds.
        self._default_conds = copy.deepcopy(self._model.conds)

        self._conds_cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max = settings.chatterbox_prompt_cache_size

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def dtype_str(self) -> str:
        return self._dtype_str

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def supports_languages(self) -> bool:
        return self._variant == "multilingual"

    @property
    def builtin_voices_list(self) -> list[str]:
        return ["default"]

    @property
    def model_id(self) -> str:
        override = (self._settings.chatterbox_model or "").strip()
        return override or _DEFAULT_REPO_PER_VARIANT[self._variant]

    def list_languages(self) -> list[tuple[str, str]]:
        if self._variant != "multilingual":
            return []
        from chatterbox import SUPPORTED_LANGUAGES

        return list(SUPPORTED_LANGUAGES.items())

    # ------------------------------------------------------------------
    # Model loading

    def _load_model(self):
        variant = self._variant
        device = self._device_str
        override = (self._settings.chatterbox_model or "").strip()

        if variant == "standard":
            from chatterbox import ChatterboxTTS as Cls
        elif variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Cls
        elif variant == "multilingual":
            from chatterbox import ChatterboxMultilingualTTS as Cls
        else:  # pragma: no cover — guarded by pydantic Literal
            raise ValueError(f"unknown chatterbox_variant: {variant}")

        # Local directory ⇒ load as-is.
        if override and os.path.isdir(override):
            log.info("loading %s from local dir %s", Cls.__name__, override)
            return Cls.from_local(override, device)

        # Otherwise resolve a HuggingFace repo id — either user-supplied or
        # the variant's default — and snapshot_download it.
        repo_id = override or _DEFAULT_REPO_PER_VARIANT[variant]
        from huggingface_hub import snapshot_download

        log.info(
            "downloading %s from HF repo %s via snapshot_download",
            Cls.__name__,
            repo_id,
        )
        ckpt_dir = snapshot_download(
            repo_id=repo_id,
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=_HF_ALLOW_PATTERNS[variant],
        )
        return Cls.from_local(ckpt_dir, device)

    # ------------------------------------------------------------------
    # Conditionals cache

    def _make_cache_key(
        self, ref_audio: str, ref_mtime: float, kwargs: dict
    ) -> tuple:
        exag = kwargs.get("exaggeration")
        if exag is None:
            exag = _default_exaggeration(self._variant)
        if self._variant == "turbo":
            norm = kwargs.get("norm_loudness")
            if norm is None:
                norm = True
            return (ref_audio, float(ref_mtime), float(exag), bool(norm))
        return (ref_audio, float(ref_mtime), float(exag))

    def _cache_get(self, key: tuple):
        with self._cache_lock:
            found = self._conds_cache.get(key)
            if found is not None:
                self._conds_cache.move_to_end(key)
        return found

    def _cache_put(self, key: tuple, conds) -> None:
        with self._cache_lock:
            self._conds_cache[key] = conds
            self._conds_cache.move_to_end(key)
            while len(self._conds_cache) > self._cache_max:
                self._conds_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Kwargs filtering

    def _filter_generate_kwargs(self, kwargs: dict) -> dict:
        allowed = _APPLICABLE_KWARGS[self._variant]
        return {k: v for k, v in kwargs.items() if v is not None and k in allowed}

    # ------------------------------------------------------------------
    # Core synchronous generate

    def _generate_sync(
        self,
        text: str,
        *,
        kind: str,
        ref_audio: Optional[str],
        ref_mtime: Optional[float],
        kwargs: dict,
    ) -> np.ndarray:
        filtered = self._filter_generate_kwargs(kwargs)

        with self._inference_lock:
            audio_prompt_path: Optional[str] = None
            cache_key: Optional[tuple] = None

            if kind == "builtin":
                # Restore the shipped default speaker (may have been overwritten
                # by a previous file-voice request).
                self._model.conds = copy.deepcopy(self._default_conds)
            else:  # clone
                assert ref_audio is not None
                if ref_mtime is not None:
                    cache_key = self._make_cache_key(ref_audio, ref_mtime, kwargs)
                    cached = self._cache_get(cache_key)
                    if cached is not None:
                        self._model.conds = copy.deepcopy(cached)
                    else:
                        audio_prompt_path = ref_audio
                else:
                    # Temp uploads (/v1/audio/clone): never cache.
                    audio_prompt_path = ref_audio

            if self._variant == "multilingual":
                lang = (
                    kwargs.get("language_id")
                    or self._default_language
                )
                tensor = self._model.generate(
                    text,
                    language_id=lang,
                    audio_prompt_path=audio_prompt_path,
                    **filtered,
                )
            else:
                tensor = self._model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    **filtered,
                )

            if (
                kind == "clone"
                and audio_prompt_path is not None
                and cache_key is not None
                and self._model.conds is not None
            ):
                try:
                    snapshot = copy.deepcopy(self._model.conds)
                    self._cache_put(cache_key, snapshot)
                except Exception:  # pragma: no cover
                    log.exception(
                        "failed to cache Conditionals for key=%r", cache_key
                    )

        # tensor shape: (1, N) float on device — convert to mono float32 np.
        return (
            tensor.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    # ------------------------------------------------------------------
    # Public async synthesis

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_mtime: Optional[float] = None,
        **kwargs: object,
    ) -> np.ndarray:
        return await asyncio.to_thread(
            self._generate_sync,
            text,
            kind="clone",
            ref_audio=ref_audio,
            ref_mtime=ref_mtime,
            kwargs=dict(kwargs),
        )

    async def synthesize_builtin(
        self,
        text: str,
        *,
        voice: str,
        **kwargs: object,
    ) -> np.ndarray:
        if voice != "default":
            raise ValueError(f"unknown builtin voice: {voice}")
        return await asyncio.to_thread(
            self._generate_sync,
            text,
            kind="builtin",
            ref_audio=None,
            ref_mtime=None,
            kwargs=dict(kwargs),
        )
