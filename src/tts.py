"""OpenAI text-to-speech integration with local audio caching."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "coral"
    model: str = "gpt-4o-mini-tts"
    stable_frames: int = 8
    cooldown_seconds: float = 1.5
    audio_dir: Path = Path("audio_cache")


def cached_audio_path(letter: str, audio_dir: Path) -> Path:
    safe_letter = letter.strip().upper()
    if len(safe_letter) != 1 or not safe_letter.isalpha():
        raise ValueError(f"Invalid letter for TTS cache: {letter!r}")
    return audio_dir / f"{safe_letter}.mp3"


def should_speak(letter: str | None, state: dict[str, Any], config: TTSConfig) -> bool:
    """Return True once a letter is stable and outside its cooldown."""
    if not letter:
        state["letter"] = None
        state["count"] = 0
        return False

    if state.get("letter") == letter:
        state["count"] = int(state.get("count", 0)) + 1
    else:
        state["letter"] = letter
        state["count"] = 1

    now = float(state.get("now", time.time()))
    last_spoken = state.setdefault("last_spoken", {})
    last_time = float(last_spoken.get(letter, 0.0))
    stable = int(state["count"]) >= config.stable_frames
    cooled_down = now - last_time >= config.cooldown_seconds

    if stable and cooled_down:
        last_spoken[letter] = now
        return True
    return False


def ensure_audio(letter: str, config: TTSConfig) -> Path | None:
    """Generate or reuse the cached OpenAI TTS audio for one letter."""
    config.audio_dir.mkdir(parents=True, exist_ok=True)
    path = cached_audio_path(letter, config.audio_dir)
    if path.exists() and path.stat().st_size > 0:
        return path

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with client.audio.speech.with_streaming_response.create(
        model=config.model,
        voice=config.voice,
        input=f"Letra {letter.upper()}",
    ) as response:
        response.stream_to_file(path)
    return path


def play_audio(path: Path | None) -> None:
    """Play an MP3 file without raising if audio output is unavailable."""
    if path is None or not path.exists():
        return

    try:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
    except Exception as exc:  # pragma: no cover - hardware dependent.
        print(f"No se pudo reproducir el audio: {exc}")
