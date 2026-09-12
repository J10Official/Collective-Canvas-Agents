"""Build WebM videos from saved PNG snapshots with an isolated FFmpeg binary."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "_video_vendor"
PINNED_PACKAGE = "imageio-ffmpeg==0.6.0"
_INSTALL_LOCK = threading.Lock()


def _ffmpeg_executable() -> str:
    """Find FFmpeg, installing a pinned binary-only helper locally if needed."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    with _INSTALL_LOCK:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        try:
            module = importlib.import_module("imageio_ffmpeg")
        except ImportError:
            VENDOR.mkdir(parents=True, exist_ok=True)
            print(f"Installing isolated video helper {PINNED_PACKAGE} into {VENDOR} ...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--target",
                    str(VENDOR),
                    PINNED_PACKAGE,
                ],
                check=True,
                timeout=300,
            )
            importlib.invalidate_caches()
            module = importlib.import_module("imageio_ffmpeg")
        return str(module.get_ffmpeg_exe())


def build_webm(
    snapshots: list[Path],
    output_path: Path,
    *,
    frame_duration_seconds: float = 0.65,
    timeout_seconds: float | None = None,
) -> Path:
    """Encode every snapshot in order and retain each for the requested duration."""
    if not snapshots or any(not path.is_file() for path in snapshots):
        raise RuntimeError("video needs at least one existing snapshot")
    if not 0.05 <= frame_duration_seconds <= 10:
        raise ValueError("frame duration must be from 0.05 to 10 seconds")
    ffmpeg = _ffmpeg_executable()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = timeout_seconds or max(120.0, len(snapshots) * frame_duration_seconds + 60.0)
    with tempfile.TemporaryDirectory(prefix="collective_canvas_frames_") as temporary_name:
        temporary = Path(temporary_name)
        for index, snapshot in enumerate(snapshots):
            shutil.copyfile(snapshot, temporary / f"frame_{index:05d}.png")
        pending = output_path.with_suffix(output_path.suffix + ".tmp.webm")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"1000/{round(frame_duration_seconds * 1000)}",
            "-i",
            str(temporary / "frame_%05d.png"),
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "30",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(pending),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg video encoding failed: {completed.stderr.strip()[-1500:]}")
        if not pending.is_file() or pending.stat().st_size < 32:
            raise RuntimeError("FFmpeg completed without creating a valid WebM")
        pending.replace(output_path)
    return output_path

