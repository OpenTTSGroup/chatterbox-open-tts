# chatterbox-open-tts

**English** · [中文](./README.zh.md)

OpenAI-compatible HTTP TTS service built on top of
[Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI.
Ships as a single CUDA container image on GHCR.

Implements the [Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec):

- `POST /v1/audio/speech` — OpenAI-compatible synthesis
- `POST /v1/audio/clone` — one-shot zero-shot cloning (multipart upload)
- `GET  /v1/audio/voices` — list file-based and built-in voices
- `GET  /v1/audio/voices/preview?id=...` — download a reference WAV
- `GET  /v1/audio/languages` — list supported languages (multilingual variant only)
- `GET  /healthz` — engine status, capabilities, concurrency snapshot

Six output formats (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`); mono
`float32` encoded server-side at 24 kHz. Voices live on disk as
`${VOICES_DIR}/<id>.{wav,txt,yml}` triples. The `.txt` transcript is
accepted for spec compliance but not read by Chatterbox — voice cloning
uses the reference waveform alone.

## Variants

Chatterbox ships three models; pick one via `CHATTERBOX_VARIANT`:

| variant | parameters | language | notes |
|---|---|---|---|
| `multilingual` (default) | 500M | 23 languages (incl. English, Chinese) | `language_id` field required on each request; default from `CHATTERBOX_DEFAULT_LANGUAGE`. |
| `turbo` | 350M | English only | Lowest latency, supports paralinguistic tags like `[cough]`, `[laugh]`. |
| `standard` | 500M | English only | Richest expressiveness control (`cfg_weight`, `exaggeration`). |

All three embed a Resemble [Perth](https://github.com/resemble-ai/Perth)
imperceptible watermark in the output.

## Quick start

```bash
mkdir -p voices cache

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/chatterbox-open-tts:latest
```

First boot downloads the model weights (~2 GB) to `/root/.cache`. Mount the
cache directory to avoid repeat downloads. `/healthz` reports
`status="loading"` until the engine is ready.

Default voice (no reference needed — the model ships with a built-in speaker
embedding):

```bash
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Chatterbox.","voice":"default","language_id":"en","response_format":"mp3"}' \
  -o out.mp3
```

Voice cloning from a disk reference:

```bash
cp ~/my-ref.wav voices/alice.wav
echo "Reference transcript (not used by Chatterbox but required by the spec)." \
  > voices/alice.txt

curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello!","voice":"file://alice","language_id":"en","response_format":"mp3"}' \
  -o clone.mp3
```

## Capabilities

| capability | value | notes |
|---|---|---|
| `clone` | `true` | zero-shot via `voice="file://..."` or `POST /v1/audio/clone` |
| `streaming` | `false` | Chatterbox has no native streaming generator; `/v1/audio/realtime` is not exposed |
| `design` | `false` | Chatterbox has no text-only voice design API |
| `languages` | depends on variant | `true` only for `CHATTERBOX_VARIANT=multilingual`; otherwise `false` and `/v1/audio/languages` returns 404 |
| `builtin_voices` | `true` | a single synthetic `default` voice backed by the model's shipped `conds.pt` |

## Environment variables

### Engine (prefixed `CHATTERBOX_`)

| variable | default | description |
|---|---|---|
| `CHATTERBOX_VARIANT` | `multilingual` | `standard` / `turbo` / `multilingual` |
| `CHATTERBOX_MODEL` | `` | HuggingFace repo id or local checkpoint directory. Empty ⇒ use the variant's default repo (`ResembleAI/chatterbox` for `standard`/`multilingual`, `ResembleAI/chatterbox-turbo` for `turbo`). Set `HF_TOKEN` for private repos. |
| `CHATTERBOX_DEVICE` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `CHATTERBOX_CUDA_INDEX` | `0` | GPU index when multiple are visible |
| `CHATTERBOX_DTYPE` | `float16` | Reported in `/healthz.dtype`; Chatterbox loads model weights in their saved precision and converts at runtime, so this is primarily informational. |
| `CHATTERBOX_DEFAULT_LANGUAGE` | `en` | Multilingual variant only: default `language_id` when the request omits it. |
| `CHATTERBOX_PROMPT_CACHE_SIZE` | `16` | LRU size for cached `Conditionals` keyed by (ref_audio, mtime, exaggeration[, norm_loudness]). |

### Service-level (no prefix)

| variable | default | description |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `VOICES_DIR` | `/voices` | scan root for file-based voices |
| `MAX_INPUT_CHARS` | `8000` | 413 above this |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | in-flight synthesis ceiling. Chatterbox is not thread-safe internally (the LRU conditionals cache assumes serialised `generate()` calls); raise with caution. |
| `MAX_QUEUE_SIZE` | `0` | 0 = unbounded queue |
| `QUEUE_TIMEOUT` | `0` | seconds; 0 = unbounded wait |
| `MAX_AUDIO_BYTES` | `20971520` | upload limit for `/v1/audio/clone` |
| `CORS_ENABLED` | `false` | `true` mounts a `CORSMiddleware` that allows any origin / method / header on every endpoint (no credentials — see the [spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md#37-cors)). Keep `false` when the service is fronted by a reverse proxy or called same-origin. |

## Compose

See [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml).

## API request parameters

GET endpoints (`/healthz`, `/v1/audio/voices`, `/v1/audio/voices/preview`,
`/v1/audio/languages`) take no body and at most a single `id` query
parameter — see the [Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)
for their response shape.

The tables below describe the POST endpoints that accept a request body. The
**Status** column uses a fixed vocabulary:

- **required** — rejected with 422 if missing.
- **supported** — accepted and consumed by Chatterbox.
- **ignored** — accepted for OpenAI compatibility; has no effect.
- **conditional** — behaviour depends on the variant or other fields.
- **extension** — Chatterbox-specific field, not part of the Open TTS spec.

### `POST /v1/audio/speech` (application/json)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `model` | string | `null` | ignored | OpenAI compatibility only; any value is accepted and discarded. |
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. Empty ⇒ 422, over limit ⇒ 413. |
| `voice` | string | — | required | `default` uses the shipped built-in speaker; `file://<id>` loads `${VOICES_DIR}/<id>.wav`. Other values ⇒ 404. |
| `response_format` | enum | `mp3` | supported | One of `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`. Global default overridden by `DEFAULT_RESPONSE_FORMAT`. |
| `speed` | float | `1.0` | ignored | Accepted for OpenAI compatibility; Chatterbox has no speed control parameter. |
| `instructions` | string \| null | `null` | ignored | Accepted for OpenAI compatibility; Chatterbox has no instruct API. |
| `language_id` | string \| null | `null` | conditional | Multilingual variant: required by the model, falls back to `CHATTERBOX_DEFAULT_LANGUAGE` when omitted. Standard/turbo: dropped silently. Unsupported code on multilingual ⇒ 422. |
| `exaggeration` | float \| null | model default | extension | Standard/multilingual: expressiveness amplifier in `[0.0, 2.0]`. Turbo: dropped (model logs a warning on non-zero values). |
| `cfg_weight` | float \| null | model default | extension | Standard/multilingual: classifier-free guidance weight in `[0.0, 1.0]`. Turbo: dropped. |
| `temperature` | float \| null | model default | extension | Sampling temperature. |
| `top_p` | float \| null | model default | extension | Nucleus sampling cutoff. |
| `min_p` | float \| null | model default | extension | Standard/multilingual only. Turbo: dropped. |
| `repetition_penalty` | float \| null | model default | extension | T3 LLM decoder repetition penalty, `[1.0, 10.0]`. |
| `top_k` | int \| null | `1000` | extension | Turbo variant only; dropped by standard/multilingual. |
| `norm_loudness` | bool \| null | `true` | extension | Turbo variant only: whether to normalise output to −27 LUFS. Included in the conditionals cache key. |

### `POST /v1/audio/clone` (multipart/form-data)

Same parameters as `/v1/audio/speech`, with these differences:

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `audio` | file | — | required | Extension must be one of `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`. Over `MAX_AUDIO_BYTES` ⇒ 413. The upload is never persisted to `${VOICES_DIR}` and never enters the conditionals cache. |
| `input` | string | — | required | Same as `/speech.input`. |
| `prompt_text` | string \| null | `null` | ignored | Accepted for OpenAI/spec compatibility; Chatterbox does not consume reference transcripts. |
| all other fields (`response_format`, `language_id`, `exaggeration`, …) | | | | Same semantics as `/speech`. |

## Voices directory

```
${VOICES_DIR}/
├── alice.wav        # required — reference audio (any length; first 10 s is used)
├── alice.txt        # required by spec — Chatterbox does not read this file
└── alice.yml        # optional — mapping rendered as metadata in /v1/audio/voices
```

The `.txt` file is required for spec compliance but its contents are not fed
to Chatterbox. You can put a short placeholder there if you want —
`echo "(unused)" > alice.txt` is enough.

## Known limitations

- **No streaming.** Chatterbox's `generate()` returns the full waveform in
  a single `torch.Tensor`; the service therefore declares
  `capabilities.streaming=false` and does not expose `/v1/audio/realtime`.
- **No speed control.** Chatterbox has no speed knob, so the `speed` field
  is accepted for OpenAI compatibility but never applied.
- **No instructions.** Likewise, `instructions` is accepted and discarded.
- **Turbo is English-only** and silently ignores `cfg_weight`,
  `exaggeration`, and `min_p` — they are stripped from the request before
  reaching the model to avoid upstream warnings.
- **Multilingual requires `language_id`** to be set on every request; the
  service falls back to `CHATTERBOX_DEFAULT_LANGUAGE` (default `en`) when
  omitted.
- **First request for a new file voice is slow** (embedding extraction);
  subsequent requests with the same `(ref_audio, mtime, exaggeration[, norm_loudness])`
  key hit the `Conditionals` LRU (`CHATTERBOX_PROMPT_CACHE_SIZE`).
- **Not thread-safe.** Chatterbox mutates `self._model.conds` during
  `generate()`, so the service serialises inference with an internal
  `threading.Lock`. Raising `MAX_CONCURRENCY` above 1 will queue requests
  in the FastAPI layer and still execute them one at a time.
- **Reference audio length.** `turbo` requires reference clips longer than
  5 seconds (upstream assertion). `standard` and `multilingual` tolerate
  shorter clips but still use the first 10 seconds for S3Gen conditioning.
