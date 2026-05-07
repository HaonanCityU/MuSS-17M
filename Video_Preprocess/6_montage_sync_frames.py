"""
Build a multi-camera montage *video* from extracted frames (3_extract_frames.py layout).

Files are named row_XXXXX_frame_YYYYYYY.jpg (CSV row + that camera's video frame).

Selection uses the reference camera's video frame number in [start_frame, end_frame].
For each hit, the matching CSV row is used to load the same sync row in all Camera folders.

Speed notes:
  - One-time per-camera index (csv_row -> jpg name); avoids scanning huge folders on every frame.
  - Optional parallel JPEG decode across cameras (ThreadPoolExecutor).
  - Faster resize (INTER_LINEAR) and lighter text (LINE_8) by default.

Output is a single MP4. Progress uses tqdm.

Dependencies: opencv-python, numpy, tqdm

Expected layout::

    frames_root/
    ├── Camera1/
    │   ├── row_00001_frame_0001000.jpg
    │   └── ...
    ├── Camera2/
    └── ...

Example (run from this folder `Video_Preprocess/`, using relative paths):
    python 6_montage_sync_frames.py \
        --frames-root ./frames/0.mp4/all \
        --output-video ./check/montage_refCam1.mp4 \
        --reference-camera 1 \
        --start-frame 17172 \
        --end-frame 17472 \
        --num-cameras 6 \
        --cols 3 \
        --rows-grid 2 \
        --fps 60

Example for "all frames" (infer min/max from reference camera):
    python 6_montage_sync_frames.py \
        --frames-root ./frames/0.mp4/all \
        --output-video ./check/montage_refCam1.mp4 \
        --reference-camera 1 \
        --num-cameras 6 \
        --cols 3 \
        --rows-grid 2 \
        --fps 60
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    print("OpenCV required: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from tqdm import tqdm
except ImportError as e:
    print("tqdm required: pip install tqdm", file=sys.stderr)
    raise SystemExit(1) from e

_ROW_FRAME_RE = re.compile(r"^row_(\d+)_frame_(\d+)\.jpg$", re.IGNORECASE)


def _parse_row_frame_jpg(name: str) -> Optional[Tuple[int, int]]:
    m = _ROW_FRAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_per_camera_row_index(
    frames_root: Path,
    num_cameras: int,
    show_progress: bool = True,
) -> List[Dict[int, str]]:
    """
    One directory listing per camera: csv_row -> one jpg basename (same rule as before: sorted first).
    """
    indexes: List[Dict[int, str]] = []
    cam_iter = range(1, num_cameras + 1)
    if show_progress:
        cam_iter = tqdm(cam_iter, desc="Index camera folders", unit="cam")

    for cam in cam_iter:
        cam_dir = frames_root / f"Camera{cam}"
        row_to_names: Dict[int, List[str]] = defaultdict(list)
        if cam_dir.is_dir():
            for f in cam_dir.iterdir():
                if not f.is_file() or f.suffix.lower() != ".jpg":
                    continue
                parsed = _parse_row_frame_jpg(f.name)
                if parsed is None:
                    continue
                csv_row, _vid = parsed
                row_to_names[csv_row].append(f.name)
        m: Dict[int, str] = {
            r: sorted(names)[0] for r, names in row_to_names.items()
        }
        indexes.append(m)
    return indexes


def map_ref_video_frames_from_index(
    row_index: List[Dict[int, str]],
    reference_camera: int,
    start_frame: int,
    end_frame: int,
    frames_root: Path,
) -> Dict[int, int]:
    """
    From the prebuilt reference-camera index (one jpg per csv_row), build
    video_frame -> csv_row for ref frames in [start_frame, end_frame].

    If multiple csv_rows map to the same ref video frame, keep the smallest csv_row.
    (Same rule as a full directory scan when there is only one file per csv_row.)
    """
    if not (1 <= reference_camera <= len(row_index)):
        raise FileNotFoundError(
            f"reference_camera {reference_camera} out of range for index length {len(row_index)}"
        )
    cam_dir = frames_root / f"Camera{reference_camera}"
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Reference camera folder not found: {cam_dir}")

    ref_map = row_index[reference_camera - 1]
    out: Dict[int, int] = {}
    for csv_row, name in ref_map.items():
        parsed = _parse_row_frame_jpg(name)
        if parsed is None:
            continue
        _, vid_frame = parsed
        if start_frame <= vid_frame <= end_frame:
            if vid_frame not in out or csv_row < out[vid_frame]:
                out[vid_frame] = csv_row
    return out


def _imread_cam(args: Tuple[Path, Optional[str]]) -> Optional[np.ndarray]:
    cam_dir, basename = args
    if not basename:
        return None
    path = cam_dir / basename
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def build_montage_for_row_indexed(
    frames_root: Path,
    csv_row: int,
    row_index: List[Dict[int, str]],
    num_cameras: int,
    cols: int,
    rows_grid: int,
    resize_interpolation: int,
    executor: Optional[concurrent.futures.ThreadPoolExecutor],
    draw_camera_labels: bool = True,
) -> np.ndarray:
    """Build one montage using prebuilt row_index; optional parallel imread."""
    if cols * rows_grid < num_cameras:
        raise ValueError("cols * rows_grid must be >= num_cameras")

    cam_dirs = [frames_root / f"Camera{i}" for i in range(1, num_cameras + 1)]
    tasks = [
        (cam_dirs[i], row_index[i].get(csv_row)) for i in range(num_cameras)
    ]

    if executor is not None:
        tiles_bgr: List[Optional[np.ndarray]] = list(executor.map(_imread_cam, tasks))
    else:
        tiles_bgr = [_imread_cam(t) for t in tasks]

    valid = [t for t in tiles_bgr if t is not None]
    if not valid:
        raise FileNotFoundError(f"csv row {csv_row}: no jpg found for any camera")

    min_w = min(t.shape[1] for t in valid)
    min_h = min(t.shape[0] for t in valid)

    resized: List[np.ndarray] = []
    for i, t in enumerate(tiles_bgr):
        if t is None:
            tile = np.full((min_h, min_w, 3), 24, dtype=np.uint8)
        else:
            tile = cv2.resize(
                t, (min_w, min_h), interpolation=resize_interpolation
            )
        if draw_camera_labels:
            cv2.putText(
                tile,
                f"Camera{i + 1}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 255, 80),
                2,
                cv2.LINE_8,
            )
        resized.append(tile)

    w, h = min_w, min_h
    row_stripes = []
    for r in range(rows_grid):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(resized):
                row_tiles.append(resized[idx])
            else:
                row_tiles.append(np.zeros((h, w, 3), dtype=np.uint8))
        row_stripes.append(np.hstack(row_tiles))
    grid = np.vstack(row_stripes)
    return grid


def main() -> None:
    def _parse_optional_int(s: Optional[str]) -> Optional[int]:
        # Allow `--start-frame none` / `--end-frame None`.
        if s is None:
            return None
        if isinstance(s, str) and s.lower() == "none":
            return None
        return int(s)

    parser = argparse.ArgumentParser(
        description="Build a multi-camera montage MP4 from extracted row_*_frame_*.jpg frames.",
    )
    parser.add_argument(
        "--frames-root",
        required=True,
        type=str,
        help="Path containing Camera1/Camera2/... folders with row_XXXXX_frame_YYYYYYY.jpg files.",
    )
    parser.add_argument(
        "--output-video",
        required=True,
        type=str,
        help="Output MP4 path (e.g. ./check/montage_refCam1.mp4).",
    )
    parser.add_argument("--reference-camera", type=int, default=1)
    parser.add_argument(
        "--start-frame",
        type=_parse_optional_int,
        default=None,
        help="Start video frame number; pass 'none' to use the earliest available in reference camera.",
    )
    parser.add_argument(
        "--end-frame",
        type=_parse_optional_int,
        default=None,
        help="End video frame number; pass 'none' to use the latest available in reference camera.",
    )
    parser.add_argument("--num-cameras", type=int, default=6)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows-grid", type=int, default=2, dest="rows_grid")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--video-fourcc", type=str, default="mp4v")

    # Speed: parallel JPEG decode per frame (usually faster on SSD/NVMe).
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument("--parallel-imread", action="store_true", dest="parallel_imread")
    parallel_group.add_argument("--no-parallel-imread", action="store_false", dest="parallel_imread")
    parser.set_defaults(parallel_imread=True)
    parser.add_argument("--imread-workers", type=int, default=6)

    # cv2.INTER_LINEAR is faster than INTER_AREA; use INTER_AREA if you need sharper downscale.
    parser.add_argument(
        "--resize-interpolation",
        choices=["linear", "area"],
        default="linear",
    )
    # Optional: scale final montage before VideoWriter (e.g. 0.5 = half size, faster encode).
    parser.add_argument("--output-scale", type=float, default=None)

    # Optional overlay labels on each tile.
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--draw-camera-labels", action="store_true", dest="draw_camera_labels")
    label_group.add_argument(
        "--no-draw-camera-labels", action="store_false", dest="draw_camera_labels"
    )
    parser.set_defaults(draw_camera_labels=True)

    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    output_video = Path(args.output_video)
    reference_camera = args.reference_camera
    start_frame = args.start_frame
    end_frame = args.end_frame
    num_cameras = args.num_cameras
    cols = args.cols
    rows_grid = args.rows_grid
    fps = args.fps
    video_fourcc = args.video_fourcc
    parallel_imread = args.parallel_imread
    imread_workers = args.imread_workers
    resize_interpolation = (
        cv2.INTER_AREA if args.resize_interpolation == "area" else cv2.INTER_LINEAR
    )
    output_scale: Optional[float] = args.output_scale
    draw_camera_labels = args.draw_camera_labels

    try:
        row_index = build_per_camera_row_index(frames_root, num_cameras, show_progress=True)

        # If user asks for "all frames", infer min/max frame from reference camera filenames.
        if not (1 <= reference_camera <= len(row_index)):
            raise FileNotFoundError(
                f"reference_camera {reference_camera} out of range for index length {len(row_index)}"
            )
        ref_names = row_index[reference_camera - 1].values()
        vid_frames: List[int] = []
        for name in ref_names:
            parsed = _parse_row_frame_jpg(name)
            if parsed is None:
                continue
            _, vid_frame = parsed
            vid_frames.append(vid_frame)

        if not vid_frames:
            raise FileNotFoundError(
                f"No valid reference frames found for Camera{reference_camera} under {frames_root}"
            )

        if start_frame is None:
            start_frame = min(vid_frames)
        if end_frame is None:
            end_frame = max(vid_frames)

        if end_frame < start_frame:
            print("Error: end_frame must be >= start_frame", file=sys.stderr)
            sys.exit(2)

        frame_to_csv_row = map_ref_video_frames_from_index(
            row_index,
            reference_camera,
            start_frame,
            end_frame,
            frames_root,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    if not frame_to_csv_row:
        print(
            f"No frames in range [{start_frame}, {end_frame}] under "
            f"Camera{reference_camera} in {frames_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    ordered = sorted(frame_to_csv_row.items(), key=lambda x: x[0])
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[cv2.VideoWriter] = None
    out_size: Optional[Tuple[int, int]] = None
    frames_written = 0

    def run_encode_loop(executor: Optional[concurrent.futures.ThreadPoolExecutor]) -> None:
        nonlocal writer, out_size, frames_written
        for vid_frame, csv_row in tqdm(
            ordered,
            desc="Montage video",
            unit="frm",
            total=len(ordered),
        ):
            try:
                grid_bgr = build_montage_for_row_indexed(
                    frames_root,
                    csv_row,
                    row_index,
                    num_cameras=num_cameras,
                    cols=cols,
                    rows_grid=rows_grid,
                    resize_interpolation=resize_interpolation,
                    executor=executor,
                    draw_camera_labels=draw_camera_labels,
                )
            except FileNotFoundError as err:
                tqdm.write(f"Skip ref_frame {vid_frame} (csv_row {csv_row}): {err}")
                continue

            if output_scale is not None and 0 < output_scale < 1.0:
                grid_bgr = cv2.resize(
                    grid_bgr,
                    (
                        max(1, int(grid_bgr.shape[1] * output_scale)),
                        max(1, int(grid_bgr.shape[0] * output_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            h, w = grid_bgr.shape[:2]
            if writer is None:
                out_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
                writer = cv2.VideoWriter(
                    str(output_video),
                    fourcc,
                    float(fps),
                    out_size,
                )
                if not writer.isOpened():
                    print(
                        f"Failed to open VideoWriter for {output_video}",
                        file=sys.stderr,
                    )
                    sys.exit(2)

            assert out_size is not None
            if grid_bgr.shape[1] != out_size[0] or grid_bgr.shape[0] != out_size[1]:
                grid_bgr = cv2.resize(
                    grid_bgr,
                    out_size,
                    interpolation=resize_interpolation,
                )
            writer.write(grid_bgr)
            frames_written += 1

    try:
        if parallel_imread:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=imread_workers
            ) as executor:
                run_encode_loop(executor)
        else:
            run_encode_loop(None)
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        if output_video.is_file():
            try:
                output_video.unlink()
            except OSError:
                pass
        print("No frames written; removed empty output if present.", file=sys.stderr)
        sys.exit(1)

    print(f"Done: {frames_written} frame(s) -> {output_video.resolve()} @ {fps} fps")


if __name__ == "__main__":
    main()
