"""Generate a labeled empty-grid PNG without third-party packages."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "111", "100", "111"),
    "3": ("110", "001", "111", "001", "110"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "110"),
    "6": ("011", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "110"),
}


def _png_bytes(width: int, height: int, pixels: bytearray) -> bytes:
    raw = b"".join(b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3] for row in range(height))

    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_grid_png(
    path: Path,
    size: int = 16,
    cell: int = 28,
    rows: list[list[str | None]] | None = None,
    colors: dict[str, str] | None = None,
) -> Path:
    """Write a labeled grid, optionally populated with the current canvas state."""
    if not 1 <= size <= 99:
        raise ValueError("labeled grid size must be from 1 to 99")
    margin, outer = 52, 12
    width = height = margin + size * cell + outer
    background, paper, line, ink = (244, 241, 233), (255, 254, 250), (181, 176, 165), (55, 54, 50)
    pixels = bytearray(background * (width * height))

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            start = (y * width + max(0, x0)) * 3
            end = (y * width + min(width, x1)) * 3
            pixels[start:end] = bytes(color) * ((end - start) // 3)

    def text(value: str, center_x: int, center_y: int, scale: int = 2) -> None:
        glyph_width = 3 * scale
        total = len(value) * glyph_width + max(0, len(value) - 1) * scale
        left, top = center_x - total // 2, center_y - (5 * scale) // 2
        for index, character in enumerate(value):
            for row, bits in enumerate(FONT[character]):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        x = left + index * (glyph_width + scale) + column * scale
                        y = top + row * scale
                        rectangle(x, y, x + scale, y + scale, ink)

    rectangle(margin, margin, margin + size * cell, margin + size * cell, paper)
    if rows is not None:
        if len(rows) != size or any(len(row) != size for row in rows):
            raise ValueError(f"pixel rows must be exactly {size}x{size}")
        if colors is None:
            raise ValueError("colors are required when pixel rows are supplied")
        for y, row in enumerate(rows):
            for x, name in enumerate(row):
                if name is None:
                    continue
                if name not in colors:
                    raise ValueError(f"unknown color in pixel rows: {name}")
                rgb = tuple(bytes.fromhex(colors[name].lstrip("#")))
                rectangle(margin + x * cell + 1, margin + y * cell + 1, margin + (x + 1) * cell, margin + (y + 1) * cell, rgb)
    for index in range(size + 1):
        position = margin + index * cell
        rectangle(position, margin, position + 1, margin + size * cell + 1, line)
        rectangle(margin, position, margin + size * cell + 1, position + 1, line)
    for index in range(size):
        center = margin + index * cell + cell // 2
        text(str(index), center, margin // 2)
        text(str(index), margin // 2, center)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height, pixels))
    return path


def write_pixel_art_png(path: Path, rows: list[list[str | None]], colors: dict[str, str], scale: int = 32) -> Path:
    """Render canvas state to a crisp PNG with fine grid lines."""
    height, width = len(rows), len(rows[0]) if rows else 0
    if not width or any(len(row) != width for row in rows):
        raise ValueError("pixel rows must form a non-empty rectangle")
    image_width, image_height = width * scale + 1, height * scale + 1
    pixels = bytearray((255, 254, 250) * (image_width * image_height))

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(y0, y1):
            start, end = (y * image_width + x0) * 3, (y * image_width + x1) * 3
            pixels[start:end] = bytes(color) * (x1 - x0)

    for y, row in enumerate(rows):
        for x, name in enumerate(row):
            if name:
                value = colors[name].lstrip("#")
                rectangle(x * scale, y * scale, (x + 1) * scale, (y + 1) * scale, tuple(bytes.fromhex(value)))
    line = (216, 211, 199)
    for index in range(width + 1):
        rectangle(index * scale, 0, index * scale + 1, image_height, line)
    for index in range(height + 1):
        rectangle(0, index * scale, image_width, index * scale + 1, line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(image_width, image_height, pixels))
    return path


if __name__ == "__main__":
    output = write_grid_png(Path(__file__).resolve().parent / "artifacts" / "empty_grid_16.png")
    print(output)
