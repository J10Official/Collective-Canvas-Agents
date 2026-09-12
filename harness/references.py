"""Built-in and user-saved reference drawings."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .grid_image import write_grid_png
from .store import COLORS, CanvasError

DEFAULT_OBJECTS = {
    "apple": {
        "name": "apple",
        "description": "For an apple, use a red body, a black stem or outline, and a green leaf. Use other allowed colors only if they improve the image.",
    },
    "orange": {
        "name": "orange",
        "description": "For an orange, use a rounded orange body, a small green leaf, and a black or green stem. Use yellow sparingly for highlights if it improves the image.",
    },
    "lemon": {
        "name": "lemon",
        "description": "For a lemon, use one compact yellow oval with slightly tapered ends, a small green leaf, and a short black stem. Use orange sparingly for highlights.",
    },
    "eggplant": {
        "name": "eggplant",
        "description": "For an eggplant, use one elongated purple body with a broad green cap and a short black stem. Use pink sparingly for highlights.",
    },
    "tree": {
        "name": "tree",
        "description": "For a tree, use a broad green canopy, an orange trunk with a black outline, and a stable visible base. Keep the canopy and trunk connected.",
    },
    "fish": {
        "name": "fish",
        "description": "For a fish, use a blue body, a yellow tail and fins, and a black outline and eye. Keep the body, tail, and head connected in one clear side-view silhouette.",
    },
    "arrow": {
        "name": "green arrow pointing up",
        "description": "For a green arrow pointing up, use one solid green upward-pointing arrow with a broad triangular head and a centered rectangular shaft. Do not add an outline.",
    },
    "balloon": {
        "name": "hot-air balloon",
        "description": "For a hot-air balloon, use a symmetrical rounded canopy with red outer panels, orange middle panels, and yellow center panels. Add two black ropes and a small orange basket with a black border.",
    },
}

REFERENCE_ROOT = Path(__file__).resolve().parent / "reference_images"
REFERENCE_INDEX = REFERENCE_ROOT / "saved_references.json"
_REFERENCE_LOCK = threading.Lock()


def _load_saved() -> dict[str, dict[str, Any]]:
    if not REFERENCE_INDEX.exists():
        return {}
    try:
        value = json.loads(REFERENCE_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_saved(value: dict[str, dict[str, Any]]) -> None:
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = REFERENCE_INDEX.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(REFERENCE_INDEX)


def reference_tag_for_subject(subject: str) -> str:
    """Return a stable, filesystem-safe tag for an object name."""
    normalized = subject.strip().casefold()
    if normalized in DEFAULT_OBJECTS:
        return normalized
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:40] or "object"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def available_objects() -> dict[str, dict[str, Any]]:
    """List built-in objects plus objects with saved user references."""
    objects = {
        tag: {**value, "reference_available": True, "reference_source": "built_in"}
        for tag, value in DEFAULT_OBJECTS.items()
    }
    with _REFERENCE_LOCK:
        saved = _load_saved()
    for tag, record in saved.items():
        objects[tag] = {
            "name": record["name"],
            "description": record.get("description", ""),
            "reference_available": True,
            "reference_source": "saved_run",
            "reference_size": record["size"],
            "source_run_id": record.get("source_run_id"),
        }
    return objects


def reference_exists(tag: str) -> bool:
    return tag in DEFAULT_OBJECTS or tag in available_objects()


def save_reference(
    *,
    subject: str,
    description: str,
    rows: list[list[str | None]],
    source_run_id: str,
) -> tuple[str, dict[str, Any]]:
    """Persist a completed canvas as the reference associated with an object."""
    size = len(rows)
    if not subject.strip() or not 8 <= size <= 64 or any(len(row) != size for row in rows):
        raise CanvasError("saved reference must have a name and a square 8 to 64 pixel canvas")
    if any(value is not None and value not in COLORS for row in rows for value in row):
        raise CanvasError("saved reference contains an unknown color")
    tag = reference_tag_for_subject(subject)
    record = {
        "name": subject.strip(),
        "description": description.strip(),
        "size": size,
        "pixels": rows,
        "source_run_id": source_run_id,
        "saved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "image": f"{tag}.png",
    }
    with _REFERENCE_LOCK:
        saved = _load_saved()
        saved[tag] = record
        write_grid_png(REFERENCE_ROOT / record["image"], size, rows=rows, colors=COLORS)
        _write_saved(saved)
    return tag, {
        "name": record["name"],
        "description": record["description"],
        "reference_available": True,
        "reference_source": "saved_run",
        "reference_size": size,
        "source_run_id": source_run_id,
    }


def _base_rows(tag: str) -> list[list[str | None]]:
    rows: list[list[str | None]] = [[None for _ in range(16)] for _ in range(16)]

    def fill(y: int, start: int, stop: int, color: str) -> None:
        for x in range(start, stop + 1):
            rows[y][x] = color

    if tag == "apple":
        rows[2][8] = "black"
        rows[3][8] = "black"
        fill(3, 9, 10, "green")
        fill(4, 5, 9, "red")
        fill(4, 10, 11, "green")
        fill(5, 4, 11, "red")
        fill(6, 3, 12, "red")
        for y in range(7, 11):
            fill(y, 3, 12, "red")
        fill(11, 4, 11, "red")
        fill(12, 5, 10, "red")
    elif tag == "orange":
        rows[2][8] = "green"
        fill(3, 8, 10, "green")
        fill(4, 6, 9, "orange")
        fill(5, 4, 11, "orange")
        fill(6, 3, 12, "orange")
        for y in range(7, 11):
            fill(y, 2, 13, "orange")
        fill(11, 3, 12, "orange")
        fill(12, 4, 11, "orange")
        fill(13, 6, 9, "orange")
        rows[6][5] = "yellow"
        rows[7][4] = "yellow"
    elif tag == "lemon":
        fill(3, 10, 11, "green")
        rows[4][8] = "black"
        fill(4, 9, 10, "green")
        for y, start, stop in (
            (5, 4, 10), (6, 3, 11), (7, 2, 12), (8, 2, 13),
            (9, 2, 13), (10, 3, 12), (11, 4, 11), (12, 5, 10),
        ):
            fill(y, start, stop, "yellow")
        rows[7][4] = "orange"
        rows[8][3] = "orange"
    elif tag == "eggplant":
        rows[1][8] = "black"
        rows[2][8] = "black"
        fill(2, 6, 7, "green")
        fill(2, 9, 10, "green")
        fill(3, 5, 10, "green")
        rows[4][5] = "green"
        rows[4][10] = "green"
        for y, start, stop in (
            (4, 6, 9), (5, 5, 10), (6, 5, 11), (7, 4, 11), (8, 4, 12),
            (9, 3, 12), (10, 3, 12), (11, 4, 11), (12, 5, 10), (13, 6, 9),
        ):
            fill(y, start, stop, "purple")
        rows[6][6] = "pink"
        rows[7][5] = "pink"
    elif tag == "tree":
        fill(1, 6, 9, "black")
        fill(2, 4, 5, "black")
        fill(2, 6, 9, "green")
        fill(2, 10, 11, "black")
        rows[3][3] = "black"
        fill(3, 4, 11, "green")
        rows[3][12] = "black"
        rows[4][2] = "black"
        fill(4, 3, 12, "green")
        rows[4][13] = "black"
        rows[5][2] = "black"
        fill(5, 3, 12, "green")
        rows[5][13] = "black"
        rows[6][3] = "black"
        fill(6, 4, 11, "green")
        rows[6][12] = "black"
        rows[7][4] = "black"
        fill(7, 5, 10, "green")
        rows[7][11] = "black"
        for y in range(8, 13):
            rows[y][6] = "black"
            fill(y, 7, 8, "orange")
            rows[y][9] = "black"
        fill(13, 5, 10, "black")
    elif tag == "fish":
        fill(4, 5, 10, "black")
        rows[5][3] = "yellow"
        rows[5][4] = "black"
        fill(5, 5, 10, "blue")
        rows[5][11] = "black"
        fill(6, 2, 3, "yellow")
        rows[6][4] = "black"
        fill(6, 5, 11, "blue")
        rows[6][12] = "black"
        fill(7, 1, 4, "yellow")
        fill(7, 5, 12, "blue")
        rows[7][10] = "black"
        rows[7][13] = "black"
        fill(8, 1, 4, "yellow")
        fill(8, 5, 12, "blue")
        rows[8][13] = "black"
        fill(9, 2, 3, "yellow")
        rows[9][4] = "black"
        fill(9, 5, 11, "blue")
        rows[9][12] = "black"
        rows[10][3] = "yellow"
        rows[10][4] = "black"
        fill(10, 5, 10, "blue")
        rows[10][11] = "black"
        fill(11, 5, 10, "black")
    elif tag == "arrow":
        # Solid green silhouette: a broad triangular head over a centered shaft.
        for y, start, stop in (
            (1, 7, 8),
            (2, 6, 9),
            (3, 5, 10),
            (4, 4, 11),
            (5, 3, 12),
            (6, 2, 13),
            (7, 2, 13),
            (8, 5, 10),
            (9, 5, 10),
            (10, 5, 10),
            (11, 5, 10),
            (12, 5, 10),
            (13, 5, 10),
            (14, 5, 10),
        ):
            fill(y, start, stop, "green")
    elif tag == "balloon":
        rows[1][6] = rows[1][9] = "black"
        fill(1, 7, 8, "yellow")
        rows[2][5] = rows[2][10] = "black"
        rows[2][6] = rows[2][9] = "orange"
        fill(2, 7, 8, "yellow")
        rows[3][4] = rows[3][11] = "black"
        rows[3][5] = rows[3][10] = "red"
        rows[3][6] = rows[3][9] = "orange"
        fill(3, 7, 8, "yellow")
        for y in range(4, 7):
            rows[y][3] = rows[y][12] = "black"
            fill(y, 4, 5, "red")
            fill(y, 6, 6, "orange")
            fill(y, 7, 8, "yellow")
            fill(y, 9, 9, "orange")
            fill(y, 10, 11, "red")
        rows[7][4] = rows[7][11] = "black"
        rows[7][5] = rows[7][10] = "red"
        rows[7][6] = rows[7][9] = "orange"
        fill(7, 7, 8, "yellow")
        rows[8][5] = rows[8][10] = "black"
        rows[8][6] = rows[8][9] = "orange"
        fill(8, 7, 8, "yellow")
        fill(9, 6, 9, "black")
        for y in range(10, 12):
            rows[y][6] = rows[y][9] = "black"
        fill(12, 5, 10, "black")
        rows[13][5] = rows[13][10] = "black"
        fill(13, 6, 9, "orange")
        fill(14, 5, 10, "black")
    else:
        raise CanvasError(f"unknown reference tag: {tag}")
    return rows


def reference_rows(tag: str, size: int) -> list[list[str | None]]:
    """Scale a tagged 16×16 target to a square grid with nearest-cell coverage."""
    if not 8 <= size <= 64:
        raise CanvasError("reference size must be from 8 to 64")
    with _REFERENCE_LOCK:
        saved = _load_saved().get(tag)
    if saved is not None:
        base = saved.get("pixels")
        source_size = saved.get("size")
        if not isinstance(base, list) or not isinstance(source_size, int):
            raise CanvasError(f"saved reference is invalid: {tag}")
    elif tag in DEFAULT_OBJECTS:
        base = _base_rows(tag)
        source_size = 16
    else:
        raise CanvasError(f"unknown reference tag: {tag}")
    rows: list[list[str | None]] = [[None for _ in range(size)] for _ in range(size)]
    for y in range(size):
        source_y = min(source_size - 1, (y * source_size) // size)
        for x in range(size):
            source_x = min(source_size - 1, (x * source_size) // size)
            rows[y][x] = base[source_y][source_x]
    return rows


def write_reference_png(path: Path, tag: str, size: int) -> Path:
    return write_grid_png(path, size, rows=reference_rows(tag, size), colors=COLORS)
