"""Thread-safe, atomically persisted state for a configurable canvas."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

WIDTH = 32
HEIGHT = 32
COLORS = {
    "black": "#292925",
    "red": "#c95f5f",
    "orange": "#d8894b",
    "yellow": "#d2b84b",
    "green": "#5b9279",
    "blue": "#557eaa",
    "purple": "#8169a5",
    "pink": "#c56f8b",
}


class CanvasError(ValueError):
    """Raised when an attempted canvas operation is invalid."""


class CanvasStore:
    def __init__(self, path: Path, width: int = WIDTH, height: int = HEIGHT) -> None:
        if not isinstance(width, int) or not isinstance(height, int) or not 1 <= width <= 256 or not 1 <= height <= 256:
            raise CanvasError("canvas width and height must be integers from 1 to 256")
        self.path = Path(path)
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self._state = self._load_or_create()

    def _blank(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "version": 0,
            "pixels": [[None for _ in range(self.width)] for _ in range(self.height)],
        }

    def _load_or_create(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            state = self._blank()
            self._write(state)
            return state
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("width") != self.width or state.get("height") != self.height:
                state = self._blank()
                self._write(state)
                return state
            self._validate_state(state)
            return state
        except (OSError, json.JSONDecodeError, CanvasError) as exc:
            raise RuntimeError(f"Invalid canvas state at {self.path}: {exc}") from exc

    def _validate_state(self, state: dict[str, Any]) -> None:
        if state.get("width") != self.width or state.get("height") != self.height:
            raise CanvasError("stored dimensions do not match configured dimensions")
        pixels = state.get("pixels")
        if not isinstance(pixels, list) or len(pixels) != self.height:
            raise CanvasError("stored pixel rows are invalid")
        for row in pixels:
            if not isinstance(row, list) or len(row) != self.width:
                raise CanvasError("stored pixel columns are invalid")
            if any(value is not None and value not in COLORS for value in row):
                raise CanvasError("stored state contains an unknown color")
        if not isinstance(state.get("version"), int) or state["version"] < 0:
            raise CanvasError("stored version is invalid")

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps({**self._state, "colors": COLORS}))

    def set_pixel(self, x: int, y: int, color: str | None, source: str = "unknown") -> dict[str, Any]:
        with self._lock:
            if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < self.width:
                raise CanvasError(f"x must be an integer from 0 to {self.width - 1}")
            if not isinstance(y, int) or isinstance(y, bool) or not 0 <= y < self.height:
                raise CanvasError(f"y must be an integer from 0 to {self.height - 1}")
            if color is not None and color not in COLORS:
                raise CanvasError(f"color must be one of: {', '.join(COLORS)}")
            previous = self._state["pixels"][y][x]
            if previous == color:
                return {
                    "type": "pixel",
                    "changed": False,
                    "version": self._state["version"],
                    "x": x,
                    "y": y,
                    "color": color,
                    "source": source,
                }
            self._state["pixels"][y][x] = color
            self._state["version"] += 1
            self._write(self._state)
            return {
                "type": "pixel",
                "changed": True,
                "version": self._state["version"],
                "x": x,
                "y": y,
                "color": color,
                "source": source,
            }

    def clear(self, source: str = "unknown") -> dict[str, Any]:
        with self._lock:
            self._state["pixels"] = [[None for _ in range(self.width)] for _ in range(self.height)]
            self._state["version"] += 1
            self._write(self._state)
            return {
                "type": "clear",
                "version": self._state["version"],
                "source": source,
            }

    def replace_pixels(self, pixels: list[list[str | None]], source: str = "unknown") -> dict[str, Any]:
        """Atomically replace the whole canvas, primarily when restoring a checkpoint."""
        if not isinstance(pixels, list) or len(pixels) != self.height:
            raise CanvasError(f"pixels must contain exactly {self.height} rows")
        copied: list[list[str | None]] = []
        for row in pixels:
            if not isinstance(row, list) or len(row) != self.width:
                raise CanvasError(f"each pixel row must contain exactly {self.width} columns")
            if any(value is not None and value not in COLORS for value in row):
                raise CanvasError("pixels contain an unknown color")
            copied.append(row[:])
        with self._lock:
            self._state["pixels"] = copied
            self._state["version"] += 1
            self._write(self._state)
            return {"type": "snapshot", "state": self.snapshot(), "source": source}

    def resize(self, size: int, source: str = "unknown") -> dict[str, Any]:
        with self._lock:
            if not isinstance(size, int) or isinstance(size, bool) or not 8 <= size <= 64:
                raise CanvasError("grid size must be an integer from 8 to 64")
            previous_version = self._state["version"]
            self.width = size
            self.height = size
            self._state = self._blank()
            self._state["version"] = previous_version + 1
            self._write(self._state)
            return {"type": "snapshot", "state": self.snapshot(), "source": source}
