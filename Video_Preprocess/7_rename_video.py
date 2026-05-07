import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
import argparse

BASE_DIR = "."
CAMERA_DIR_PREFIX = "camera"  # Matches Camera1/Camera2/... (case-insensitive)
EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
DRY_RUN = False  # Set True to preview mapping; False to apply renames.

# Frame counting strategy:
# - "metadata": prefer nb_frames (fast but may be missing/inaccurate for some files)
# - "count_frames": force decode-count via ffprobe (accurate but may be slow)
FRAME_COUNT_MODE = "metadata"

# ffprobe timeout in seconds. Frame counting can take long on large files.
FFPROBE_TIMEOUT_S = 120


def natural_key(p: Path):
    # Prefer numeric stems (0.mp4, 1.mp4, ...) otherwise natural name sort.
    stem = p.stem
    if re.fullmatch(r"\d+", stem):
        return (0, int(stem))
    parts = re.split(r"(\d+)", p.name)
    parts = [int(x) if x.isdigit() else x.lower() for x in parts]
    return (1, parts)


def ffprobe_nb_frames(path: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(
        cmd, text=True, stderr=subprocess.STDOUT, timeout=FFPROBE_TIMEOUT_S
    ).strip()
    if not out or out.upper() == "N/A":
        return None
    try:
        n = int(out)
    except ValueError:
        return None
    return n if n > 0 else None


def ffprobe_count_frames(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(
        cmd, text=True, stderr=subprocess.STDOUT, timeout=FFPROBE_TIMEOUT_S
    ).strip()
    if not out or out.upper() == "N/A":
        raise RuntimeError("ffprobe returned empty nb_read_frames")
    n = int(out)
    if n <= 0:
        raise RuntimeError("ffprobe returned non-positive nb_read_frames")
    return n


def opencv_frames(path: Path) -> int:
    try:
        import cv2  # pip install opencv-python
    except Exception as e:
        raise RuntimeError("OpenCV not available. Install: pip install opencv-python") from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0:
        raise RuntimeError("OpenCV returned non-positive frame count")
    return frames


def get_frames(path: Path, frame_count_mode: str) -> int:
    # 1) Fast path: ffprobe metadata nb_frames
    if frame_count_mode == "metadata":
        try:
            n = ffprobe_nb_frames(path)
            if n is not None:
                return n
        except Exception:
            pass

        # 2) Fallback: OpenCV metadata
        try:
            return opencv_frames(path)
        except Exception:
            pass

        # 3) Last resort: decode-count via ffprobe (slow)
        return ffprobe_count_frames(path)

    # Force decode-count.
    if frame_count_mode == "count_frames":
        return ffprobe_count_frames(path)

    raise ValueError(f"Unknown frame_count_mode: {frame_count_mode}")


def process_one_camera_dir(d: Path, dry_run: bool, frame_count_mode: str) -> None:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print(f"No video files found in {d}")
        return

    files.sort(key=natural_key)

    # Count frames per video
    frame_counts = {}
    for p in files:
        n = get_frames(p, frame_count_mode=frame_count_mode)
        frame_counts[p] = n
        print(f"Frames: {n:>10}  File: {p.name}")

    # Build target names: cumulative frame offset, rename to <cum>.<ext>
    mapping = []
    cum = 0
    used_targets = set()
    for p in files:
        target = p.with_name(f"{cum}{p.suffix.lower()}")
        if target in used_targets:
            raise RuntimeError(f"Target name collision: {target.name}")
        used_targets.add(target)
        mapping.append((p, target, frame_counts[p], cum))
        cum += frame_counts[p]

    print("\nPlanned renames:")
    for src, dst, n, start in mapping:
        print(f"{src.name}  ->  {dst.name}   (start={start}, frames={n})")

    if dry_run:
        print("\nDRY_RUN=True: no files renamed. Re-run without --dry-run to apply.")
        return

    # Two-phase rename to avoid collisions: temp name then final name.
    temp_map = []
    for src, dst, *_ in mapping:
        tmp = src.with_name(f".__tmp__{uuid.uuid4().hex}{src.suffix.lower()}")
        os.replace(src, tmp)
        temp_map.append((tmp, dst))

    for tmp, dst in temp_map:
        os.replace(tmp, dst)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename videos under Camera* folders to cumulative-frame-offset filenames.",
    )
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--camera-prefix", type=str, default=CAMERA_DIR_PREFIX)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the rename mapping without modifying files.",
    )
    parser.add_argument("--frame-count-mode", choices=["metadata", "count_frames"], default=FRAME_COUNT_MODE)
    args = parser.parse_args()

    camera_dir_prefix = args.camera_prefix
    dry_run = bool(args.dry_run)
    frame_count_mode = args.frame_count_mode

    base = Path(args.base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"base dir not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"base dir is not a directory: {base}")

    camera_dirs = [
        p
        for p in base.iterdir()
        if p.is_dir() and p.name.lower().startswith(camera_dir_prefix.lower())
    ]
    camera_dirs.sort(key=natural_key)

    if not camera_dirs:
        print(f"No camera dirs found in {base} (prefix={camera_dir_prefix})")
        return

    for d in camera_dirs:
        print(f"\n=== Processing {d.name} ===")
        process_one_camera_dir(d, dry_run=dry_run, frame_count_mode=frame_count_mode)


if __name__ == "__main__":
    main()