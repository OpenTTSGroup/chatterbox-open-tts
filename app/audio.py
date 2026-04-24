from __future__ import annotations

import io

import numpy as np


CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}

_PYAV_CONTAINER_FORMAT = {"mp3": "mp3", "opus": "ogg", "aac": "adts"}
_PYAV_CODEC = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}


def _normalize(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False)
    np.clip(arr, -1.0, 1.0, out=arr)
    return arr


def _to_pcm16_bytes(samples: np.ndarray) -> bytes:
    scaled = np.clip(samples * 32767.0, -32768.0, 32767.0)
    return scaled.astype("<i2", copy=False).tobytes()


def _encode_soundfile(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    elif fmt == "flac":
        sf.write(buf, samples, sample_rate, format="FLAC")
    else:  # pragma: no cover - caller guards format
        raise ValueError(f"soundfile cannot encode {fmt}")
    return buf.getvalue()


def _encode_pyav(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=_PYAV_CONTAINER_FORMAT[fmt])
    try:
        stream = container.add_stream(_PYAV_CODEC[fmt], rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


def encode(samples: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode mono float32 samples into the requested container/codec."""
    if fmt not in CONTENT_TYPES:
        raise ValueError(f"unsupported response_format: {fmt}")
    arr = _normalize(samples)
    if fmt == "pcm":
        return _to_pcm16_bytes(arr), CONTENT_TYPES[fmt]
    if fmt in ("wav", "flac"):
        return _encode_soundfile(arr, sample_rate, fmt), CONTENT_TYPES[fmt]
    return _encode_pyav(arr, sample_rate, fmt), CONTENT_TYPES[fmt]
