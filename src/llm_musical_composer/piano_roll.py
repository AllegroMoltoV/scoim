"""追加依存なしで校正用のピアノロール PNG を描画する。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

from llm_musical_composer.closure_calibration import Performance

RGB = tuple[int, int, int]


def _rectangle(
    pixels: bytearray,
    width: int,
    height: int,
    bounds: tuple[int, int, int, int],
    color: RGB,
) -> None:
    left, top, right, bottom = bounds
    for y in range(max(0, top), min(height, bottom)):
        for x in range(max(0, left), min(width, right)):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)


def _chunk(name: bytes, data: bytes) -> bytes:
    body = name + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _write_png(path: Path, pixels: bytearray, width: int, height: int) -> None:
    raw = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3]) for row in range(height)
    )
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, level=9))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def render_pair_piano_roll(
    first: Performance,
    second: Performance,
    path: Path,
    *,
    width: int = 1_200,
    height: int = 520,
) -> dict[str, Any]:
    """A と B を同じ音高範囲で左右に描く。"""
    if width < 320 or height < 200:
        raise ValueError("canvas must be at least 320 by 200 pixels")
    notes = (*first.notes, *second.notes)
    if not notes:
        raise ValueError("at least one pitched note is required")
    pitch_min = min(note.pitch for note in notes)
    pitch_max = max(note.pitch for note in notes)
    pixels = bytearray((248, 248, 248) * (width * height))
    margin = 24
    gap = 24
    panel_width = (width - margin * 2 - gap) // 2
    panel_height = height - margin * 2
    metadata: dict[str, Any] = {
        "width": width,
        "height": height,
        "pitch_range": [pitch_min, pitch_max],
        "samples": {},
    }
    for panel_index, (label, performance) in enumerate((("A", first), ("B", second))):
        left = margin + panel_index * (panel_width + gap)
        right = left + panel_width
        top = margin
        bottom = top + panel_height
        _rectangle(pixels, width, height, (left, top, right, bottom), (255, 255, 255))
        for division in range(1, 16):
            x = left + round(panel_width * division / 16)
            _rectangle(pixels, width, height, (x, top, x + 1, bottom), (225, 225, 225))
        for pedal in performance.pedals:
            if pedal.value < 64:
                continue
            x = left + round(panel_width * pedal.at_ms / max(1, performance.duration_ms))
            _rectangle(
                pixels,
                width,
                height,
                (x, bottom - 5, min(right, x + 3), bottom),
                (70, 150, 90),
            )
        for note in performance.notes:
            x1 = left + round(panel_width * note.onset_ms / max(1, performance.duration_ms))
            x2 = left + round(
                panel_width * (note.onset_ms + note.duration_ms) / max(1, performance.duration_ms)
            )
            pitch_position = (note.pitch - pitch_min) / max(1, pitch_max - pitch_min)
            y = bottom - 7 - round((panel_height - 14) * pitch_position)
            intensity = round(80 + note.velocity / 127 * 150)
            _rectangle(
                pixels,
                width,
                height,
                (x1, y - 2, max(x1 + 1, x2), y + 3),
                (45, 95, intensity),
            )
        marker = (190, 55, 55) if label == "A" else (55, 90, 190)
        _rectangle(pixels, width, height, (left, 5, left + 14, 19), marker)
        metadata["samples"][label] = {
            "note_rectangles": len(performance.notes),
            "duration_ms": performance.duration_ms,
            "panel": [left, top, right, bottom],
        }
    _write_png(Path(path), pixels, width, height)
    return metadata
