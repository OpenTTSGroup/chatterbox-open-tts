from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

FILE_VOICE_PREFIX = "file://"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    id: str
    wav_path: Path
    txt_path: Path
    yml_path: Optional[Path]
    prompt_text: str
    metadata: Optional[dict[str, Any]]
    mtime: float

    @property
    def uri(self) -> str:
        return f"{FILE_VOICE_PREFIX}{self.id}"


def _strip_prefix(voice_id: str) -> str:
    if voice_id.startswith(FILE_VOICE_PREFIX):
        return voice_id[len(FILE_VOICE_PREFIX) :]
    return voice_id


class VoiceCatalog:
    """Scan ``root`` for ``<id>.wav`` + ``<id>.txt`` [+ ``<id>.yml``] triples."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def scan(self) -> dict[str, Voice]:
        if not self._root.exists() or not self._root.is_dir():
            return {}

        by_stem: dict[str, dict[str, Path]] = {}
        for entry in self._root.iterdir():
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in (".wav", ".txt", ".yml", ".yaml"):
                continue
            key = ".yml" if ext == ".yaml" else ext
            by_stem.setdefault(entry.stem, {})[key] = entry

        voices: dict[str, Voice] = {}
        for stem, parts in by_stem.items():
            wav = parts.get(".wav")
            txt = parts.get(".txt")
            if wav is None or txt is None:
                log.warning("voice %r skipped: missing .wav or .txt", stem)
                continue
            try:
                if wav.stat().st_size == 0 or txt.stat().st_size == 0:
                    log.warning("voice %r skipped: empty .wav or .txt", stem)
                    continue
            except OSError as exc:
                log.warning("voice %r skipped: stat failed (%s)", stem, exc)
                continue

            try:
                prompt_text = txt.read_text(encoding="utf-8-sig").strip()
            except OSError as exc:
                log.warning("voice %r skipped: cannot read .txt (%s)", stem, exc)
                continue

            yml = parts.get(".yml")
            metadata: Optional[dict[str, Any]] = None
            if yml is not None:
                try:
                    parsed = yaml.safe_load(yml.read_text(encoding="utf-8-sig"))
                except (OSError, yaml.YAMLError) as exc:
                    log.warning("voice %r: .yml parse failed (%s)", stem, exc)
                else:
                    if isinstance(parsed, dict):
                        metadata = parsed
                    elif parsed is not None:
                        log.warning(
                            "voice %r: .yml top-level is not a mapping (got %s)",
                            stem,
                            type(parsed).__name__,
                        )

            mtimes = [wav.stat().st_mtime, txt.stat().st_mtime]
            if yml is not None:
                try:
                    mtimes.append(yml.stat().st_mtime)
                except OSError:
                    pass
            mtime = max(mtimes)

            voices[stem] = Voice(
                id=stem,
                wav_path=wav,
                txt_path=txt,
                yml_path=yml,
                prompt_text=prompt_text,
                metadata=metadata,
                mtime=mtime,
            )
        return voices

    def get(self, voice_id: str) -> Optional[Voice]:
        return self.scan().get(_strip_prefix(voice_id))

    def list(self) -> list[Voice]:
        return sorted(self.scan().values(), key=lambda v: v.id)
