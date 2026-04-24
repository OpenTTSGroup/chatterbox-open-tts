from __future__ import annotations

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response

from app.audio import CONTENT_TYPES, encode
from app.concurrency import ConcurrencyLimiter
from app.config import Settings, get_settings
from app.schemas import (
    Capabilities,
    HealthResponse,
    Language,
    LanguagesResponse,
    ResponseFormat,
    SpeechRequest,
    VoiceInfo,
    VoiceListResponse,
)
from app.voices import FILE_VOICE_PREFIX, Voice, VoiceCatalog

log = logging.getLogger(__name__)

CLONE_AUDIO_EXTS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm"}
)


def _build_capabilities(settings: Settings) -> Capabilities:
    return Capabilities(
        clone=True,
        streaming=False,
        design=False,
        languages=settings.chatterbox_variant == "multilingual",
        builtin_voices=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.limiter = ConcurrencyLimiter(
        settings.max_concurrency,
        settings.max_queue_size,
        settings.queue_timeout,
    )
    app.state.capabilities = _build_capabilities(settings)
    app.state.engine = None

    # Defer heavy import so module-level import of this file stays cheap
    # (FastAPI reflection, reverse-proxy smoke tests, etc.).
    from app.engine import ChatterboxEngine

    try:
        engine = ChatterboxEngine(settings)
    except Exception:
        log.exception("failed to load Chatterbox engine")
        raise

    app.state.engine = engine
    log.info(
        "engine ready: variant=%s model=%s device=%s dtype=%s sample_rate=%d",
        settings.chatterbox_variant,
        engine.model_id,
        engine.device,
        engine.dtype_str,
        engine.sample_rate,
    )

    yield


app = FastAPI(title="chatterbox-open-tts", version="1.0.0", lifespan=lifespan)

if get_settings().cors_enabled:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Helpers


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _engine(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="engine loading")
    return engine


def _limiter(request: Request) -> ConcurrencyLimiter:
    return request.app.state.limiter


def _capabilities(request: Request) -> Capabilities:
    return request.app.state.capabilities


def _catalog(request: Request) -> VoiceCatalog:
    return request.app.state.catalog


def _resolve_format(fmt: Optional[str], settings: Settings) -> str:
    chosen = fmt or settings.default_response_format
    if chosen not in CONTENT_TYPES:
        raise HTTPException(
            status_code=422, detail=f"unsupported response_format: {chosen}"
        )
    return chosen


def _validate_text(text: str, limit: int) -> None:
    if len(text) == 0:
        raise HTTPException(status_code=422, detail="input must not be empty")
    if len(text) > limit:
        raise HTTPException(status_code=413, detail=f"input exceeds {limit} chars")


def _resolve_voice(voice: str, request: Request) -> tuple[str, Optional[Voice]]:
    if voice.startswith(FILE_VOICE_PREFIX):
        found = _catalog(request).get(voice)
        if found is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
        return "clone", found

    for scheme in ("http://", "https://", "s3://"):
        if voice.startswith(scheme):
            raise HTTPException(
                status_code=501, detail="remote voice URIs not supported"
            )

    engine = _engine(request)
    if voice not in engine.builtin_voices_list:
        raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
    return "builtin", None


def _request_extensions(req: SpeechRequest) -> dict:
    """Pull Chatterbox-specific extension fields off the request as a dict."""
    return {
        "exaggeration": req.exaggeration,
        "cfg_weight": req.cfg_weight,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "min_p": req.min_p,
        "repetition_penalty": req.repetition_penalty,
        "top_k": req.top_k,
        "norm_loudness": req.norm_loudness,
        "language_id": req.language_id,
    }


# ---------------------------------------------------------------------------
# Endpoints


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = _settings(request)
    engine = request.app.state.engine
    caps = _capabilities(request)
    limiter = _limiter(request)

    if engine is None:
        return HealthResponse(
            status="loading",
            model=settings.chatterbox_model or "",
            sample_rate=0,
            capabilities=caps,
            concurrency=limiter.snapshot(),
        )

    return HealthResponse(
        status="ok",
        model=engine.model_id,
        sample_rate=engine.sample_rate,
        capabilities=caps,
        device=engine.device,
        dtype=engine.dtype_str,
        concurrency=limiter.snapshot(),
    )


@app.get("/v1/audio/voices", response_model=VoiceListResponse)
async def list_voices(request: Request) -> VoiceListResponse:
    caps = _capabilities(request)
    engine = request.app.state.engine

    voices: list[VoiceInfo] = []

    if caps.builtin_voices and engine is not None:
        for spk in engine.builtin_voices_list:
            voices.append(
                VoiceInfo(id=spk, preview_url=None, prompt_text=None, metadata=None)
            )

    for v in _catalog(request).list():
        voices.append(
            VoiceInfo(
                id=v.uri,
                preview_url=f"/v1/audio/voices/preview?id={quote(v.id, safe='')}",
                prompt_text=v.prompt_text,
                metadata=v.metadata,
            )
        )

    return VoiceListResponse(voices=voices)


@app.get("/v1/audio/voices/preview")
async def voice_preview(id: str, request: Request) -> FileResponse:
    voice = _catalog(request).get(id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
    return FileResponse(
        voice.wav_path,
        media_type="audio/wav",
        filename=f"{voice.id}.wav",
    )


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request) -> Response:
    settings = _settings(request)
    engine = _engine(request)

    _validate_text(req.input, settings.max_input_chars)
    fmt = _resolve_format(req.response_format, settings)
    kind, voice_obj = _resolve_voice(req.voice, request)
    extensions = _request_extensions(req)

    async with _limiter(request).acquire():
        try:
            if kind == "clone":
                assert voice_obj is not None
                samples = await engine.synthesize_clone(
                    req.input,
                    ref_audio=str(voice_obj.wav_path),
                    ref_mtime=voice_obj.mtime,
                    **extensions,
                )
            else:
                samples = await engine.synthesize_builtin(
                    req.input,
                    voice=req.voice,
                    **extensions,
                )
        except HTTPException:
            raise
        except ValueError as exc:
            # e.g. ChatterboxMultilingualTTS raises ValueError on unsupported language_id
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            log.exception("inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

        try:
            body, ctype = encode(samples, engine.sample_rate, fmt)
        except Exception as exc:
            log.exception("encoding failed")
            raise HTTPException(status_code=500, detail=f"encoding failed: {exc}")

    return Response(content=body, media_type=ctype)


@app.post("/v1/audio/clone")
async def clone(
    request: Request,
    audio: UploadFile = File(...),
    input: str = Form(...),
    response_format: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),  # noqa: ARG001 — accepted for compat
    speed: float = Form(1.0),  # noqa: ARG001 — accepted for compat, ignored
    instructions: Optional[str] = Form(None),  # noqa: ARG001 — accepted for compat
    model: Optional[str] = Form(None),  # noqa: ARG001 — accepted for compat
    # Chatterbox extensions ----------------------------------------------
    language_id: Optional[str] = Form(None),
    exaggeration: Optional[float] = Form(None),
    cfg_weight: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    min_p: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    norm_loudness: Optional[bool] = Form(None),
) -> Response:
    settings = _settings(request)
    engine = _engine(request)

    _validate_text(input, settings.max_input_chars)
    fmt = _resolve_format(response_format, settings)

    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    if suffix not in CLONE_AUDIO_EXTS:
        raise HTTPException(
            status_code=415, detail=f"audio format not supported: {suffix}"
        )

    tmp_dir = Path(tempfile.gettempdir()) / "chatterbox-open-tts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    size = 0
    try:
        with tmp.open("wb") as dest:
            while True:
                chunk = await audio.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_audio_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"audio exceeds {settings.max_audio_bytes} bytes",
                    )
                dest.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="audio file is empty")

        extensions = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "norm_loudness": norm_loudness,
            "language_id": language_id,
        }

        async with _limiter(request).acquire():
            try:
                samples = await engine.synthesize_clone(
                    input,
                    ref_audio=str(tmp),
                    ref_mtime=None,  # one-shot upload — never cache
                    **extensions,
                )
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            except Exception as exc:
                log.exception("clone inference failed")
                raise HTTPException(
                    status_code=500, detail=f"inference failed: {exc}"
                )

            try:
                body, ctype = encode(samples, engine.sample_rate, fmt)
            except Exception as exc:
                log.exception("clone encoding failed")
                raise HTTPException(
                    status_code=500, detail=f"encoding failed: {exc}"
                )

        return Response(content=body, media_type=ctype)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            log.warning("failed to unlink temp file %s", tmp)


# /v1/audio/languages — registered only when the multilingual variant is loaded.
if get_settings().chatterbox_variant == "multilingual":

    @app.get("/v1/audio/languages", response_model=LanguagesResponse)
    async def list_languages(request: Request) -> LanguagesResponse:
        engine = _engine(request)
        return LanguagesResponse(
            languages=[Language(key=k, name=v) for k, v in engine.list_languages()]
        )
