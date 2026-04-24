# chatterbox-open-tts

[English](./README.md) · **中文**

基于 Resemble AI 的 [Chatterbox](https://github.com/resemble-ai/chatterbox)
构建的 OpenAI 兼容 HTTP TTS 服务，单镜像发布到 GHCR。

遵循 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec)：

- `POST /v1/audio/speech` — OpenAI 兼容文本合成
- `POST /v1/audio/clone` — 一次性上传音频做零样本克隆
- `GET  /v1/audio/voices` — 列出内置音色与文件克隆音色
- `GET  /v1/audio/voices/preview?id=...` — 下载参考音频
- `GET  /v1/audio/languages` — 列出支持的语言（仅 multilingual 变体）
- `GET  /healthz` — 引擎状态、能力矩阵、并发快照

支持 `mp3`、`opus`、`aac`、`flac`、`wav`、`pcm` 六种输出格式，服务端以
24 kHz 单声道 `float32` 合成后再编码。音色目录通过
`${VOICES_DIR}/<id>.{wav,txt,yml}` 三件套提供；`.txt` 转录文件为规范要
求而保留，Chatterbox 不实际读取其内容。

## 变体

Chatterbox 提供三个模型，通过 `CHATTERBOX_VARIANT` 切换：

| 变体 | 参数量 | 语言 | 说明 |
|---|---|---|---|
| `multilingual`（默认） | 500M | 23 语言（含英语、中文） | 每次请求需要 `language_id` 字段；缺省时使用 `CHATTERBOX_DEFAULT_LANGUAGE`。 |
| `turbo` | 350M | 仅英语 | 延迟最低；支持 `[cough]`、`[laugh]` 等副语言标签。 |
| `standard` | 500M | 仅英语 | 表达控制最丰富（`cfg_weight`、`exaggeration`）。 |

三者都会在输出音频中嵌入 Resemble 的
[Perth](https://github.com/resemble-ai/Perth) 不可察觉水印。

## 快速开始

```bash
mkdir -p voices cache

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/chatterbox-open-tts:latest
```

首次启动会下载约 2 GB 权重到 `/root/.cache`；挂载 cache 目录避免重复
下载。引擎加载期间 `/healthz` 返回 `status="loading"`。

默认音色合成（无需任何参考音频，模型自带内置 speaker embedding）：

```bash
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，来自 Chatterbox。","voice":"default","language_id":"zh","response_format":"mp3"}' \
  -o out.mp3
```

基于磁盘参考音频做克隆：

```bash
cp ~/my-ref.wav voices/alice.wav
echo "参考音频对应的转录（Chatterbox 不使用此文件，仅为规范要求）。" \
  > voices/alice.txt

curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好！","voice":"file://alice","language_id":"zh","response_format":"mp3"}' \
  -o clone.mp3
```

## 能力矩阵

| capability | 取值 | 说明 |
|---|---|---|
| `clone` | `true` | 通过 `voice="file://..."` 或 `POST /v1/audio/clone` 做零样本克隆 |
| `streaming` | `false` | Chatterbox 无原生流式接口；不暴露 `/v1/audio/realtime` |
| `design` | `false` | Chatterbox 没有文本化 voice 设计 API |
| `languages` | 取决于变体 | 仅 `CHATTERBOX_VARIANT=multilingual` 时为 `true`；其他变体下 `/v1/audio/languages` 返回 404 |
| `builtin_voices` | `true` | 暴露单个 `default` 合成音色，对应模型自带的 `conds.pt` speaker embedding |

## 环境变量

### 引擎（带 `CHATTERBOX_` 前缀）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHATTERBOX_VARIANT` | `multilingual` | `standard` / `turbo` / `multilingual` |
| `CHATTERBOX_MODEL` | `` | HuggingFace repo id 或本地 checkpoint 目录。空值使用各 variant 的默认 repo（`standard`/`multilingual` 为 `ResembleAI/chatterbox`，`turbo` 为 `ResembleAI/chatterbox-turbo`）。 |
| `CHATTERBOX_DEVICE` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `CHATTERBOX_CUDA_INDEX` | `0` | 多卡时指定 GPU 序号 |
| `CHATTERBOX_DTYPE` | `float16` | 体现在 `/healthz.dtype`；Chatterbox 以权重保存时的精度加载并在运行时转换，此字段主要用于上报。 |
| `CHATTERBOX_DEFAULT_LANGUAGE` | `en` | 仅 multilingual 变体生效：请求未传 `language_id` 时的兜底值。 |
| `CHATTERBOX_PROMPT_CACHE_SIZE` | `16` | 以 (ref_audio, mtime, exaggeration[, norm_loudness]) 为 key 的 `Conditionals` LRU 缓存大小。 |

### 服务级（无前缀）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `VOICES_DIR` | `/voices` | 文件克隆音色扫描根 |
| `MAX_INPUT_CHARS` | `8000` | 超出返回 413 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | 同时推理上限。Chatterbox 内部非线程安全（conds 缓存假设串行调用 `generate()`），谨慎提高。 |
| `MAX_QUEUE_SIZE` | `0` | 0 = 不限 |
| `QUEUE_TIMEOUT` | `0` | 秒；0 = 不限 |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` 上传大小限制 |
| `CORS_ENABLED` | `false` | 设为 `true` 挂载 `CORSMiddleware`，对**所有端点**放开任意 origin / method / header（不带凭证，详见[规范 §3.7](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md#37-cors)）。反向代理前置或同源调用时保持 `false`。 |

## Compose

参考 [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml)。

## 请求参数

GET 端点（`/healthz`、`/v1/audio/voices`、`/v1/audio/voices/preview`、
`/v1/audio/languages`）无请求体，最多一个 `id` 查询参数；响应结构参见
[Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)。

下表描述有请求体的 POST 端点。**状态**列使用固定词汇：

- **required** — 必填，缺失返回 422。
- **supported** — 可选字段，引擎实际消费。
- **ignored** — 为 OpenAI 兼容接受，但永远不生效。
- **conditional** — 行为取决于变体或其他字段。
- **extension** — Chatterbox 特有扩展，规范未定义。

### `POST /v1/audio/speech`（application/json）

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容；任意值被接受后丢弃。 |
| `input` | string | — | required | 长度 1..`MAX_INPUT_CHARS`；空串 ⇒ 422，超长 ⇒ 413。 |
| `voice` | string | — | required | `default` 使用模型自带内置音色；`file://<id>` 加载 `${VOICES_DIR}/<id>.wav`。其他取值 ⇒ 404。 |
| `response_format` | enum | `mp3` | supported | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` 六选一；全局默认由 `DEFAULT_RESPONSE_FORMAT` 覆盖。 |
| `speed` | float | `1.0` | ignored | 为 OpenAI 兼容接受；Chatterbox 没有速度控制参数。 |
| `instructions` | string \| null | `null` | ignored | 为 OpenAI 兼容接受；Chatterbox 没有 instruct API。 |
| `language_id` | string \| null | `null` | conditional | multilingual 变体：模型要求此字段，缺省时回退到 `CHATTERBOX_DEFAULT_LANGUAGE`；standard/turbo 变体下静默丢弃。multilingual 下不支持的语言码 ⇒ 422。 |
| `exaggeration` | float \| null | 模型默认 | extension | standard/multilingual：表现力放大器，范围 `[0.0, 2.0]`。turbo：丢弃（模型对非零值会记 warning）。 |
| `cfg_weight` | float \| null | 模型默认 | extension | standard/multilingual：classifier-free guidance 权重，范围 `[0.0, 1.0]`。turbo：丢弃。 |
| `temperature` | float \| null | 模型默认 | extension | 采样温度。 |
| `top_p` | float \| null | 模型默认 | extension | nucleus 采样阈值。 |
| `min_p` | float \| null | 模型默认 | extension | 仅 standard/multilingual；turbo 丢弃。 |
| `repetition_penalty` | float \| null | 模型默认 | extension | T3 LLM 解码器重复惩罚，范围 `[1.0, 10.0]`。 |
| `top_k` | int \| null | `1000` | extension | 仅 turbo 变体；standard/multilingual 丢弃。 |
| `norm_loudness` | bool \| null | `true` | extension | 仅 turbo：是否归一化到 −27 LUFS。参与 conditionals 缓存 key。 |

### `POST /v1/audio/clone`（multipart/form-data）

参数与 `/v1/audio/speech` 基本一致，差异如下：

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | file | — | required | 扩展名须属于 `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`；超过 `MAX_AUDIO_BYTES` ⇒ 413。上传**不会**持久化到 `${VOICES_DIR}`，也不进 conditionals 缓存。 |
| `input` | string | — | required | 同 `/speech.input`。 |
| `prompt_text` | string \| null | `null` | ignored | 为 OpenAI / 规范兼容接受；Chatterbox 不消费参考转录。 |
| 其余字段（`response_format`、`language_id`、`exaggeration` …） | | | | 语义与 `/speech` 一致。 |

## 音色目录

```
${VOICES_DIR}/
├── alice.wav        # 必需 — 参考音频（任意长度，前 10 秒被使用）
├── alice.txt        # 规范要求 — Chatterbox 不读取该文件
└── alice.yml        # 可选 — 作为 metadata 透传到 /v1/audio/voices
```

`.txt` 文件是规范硬性要求，但 Chatterbox 不会把内容喂给模型。放一个
占位字符串即可（例如 `echo "(unused)" > alice.txt`）。

## 已知限制

- **无流式**。Chatterbox 的 `generate()` 一次性返回完整 `torch.Tensor`，
  因此声明 `capabilities.streaming=false`，不暴露 `/v1/audio/realtime`。
- **无速度控制**。Chatterbox 没有速度旋钮，`speed` 字段仅用于 OpenAI
  兼容，不会生效。
- **无 instructions 支持**。同理，`instructions` 字段被接受后丢弃。
- **Turbo 仅英语**，且会静默忽略 `cfg_weight`、`exaggeration`、`min_p` —
  服务端在转发给模型前就会剥离这三个字段，避免上游打 warning。
- **Multilingual 需要 `language_id`**，每次请求都要传；缺省时回退到
  `CHATTERBOX_DEFAULT_LANGUAGE`（默认 `en`）。
- **新文件音色首次请求较慢**（需要抽取 embedding）；同一
  `(ref_audio, mtime, exaggeration[, norm_loudness])` 的后续请求会命中
  `Conditionals` LRU（由 `CHATTERBOX_PROMPT_CACHE_SIZE` 控制）。
- **非线程安全**。Chatterbox 在 `generate()` 过程中会修改
  `self._model.conds`，因此服务端通过内部 `threading.Lock` 串行化推理。
  即使把 `MAX_CONCURRENCY` 调大，请求也只是在 FastAPI 层排队，执行仍是
  一次一个。
- **参考音频长度**。`turbo` 变体要求参考片段长于 5 秒（上游硬断言）；
  `standard` / `multilingual` 对更短的片段容忍，但只使用前 10 秒做
  S3Gen 条件编码。
