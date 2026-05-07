"""
Fast Frames to Video Converter

This script is an accelerated replacement for 4_frames2video.py.
It keeps the same filename parsing/sync logic, but uses ffmpeg for encoding.
When ffmpeg is not available, it falls back to OpenCV.

Input frame filename format:
    row_XXXXX_frame_YYYYYYY.jpg

python 4_frames2video_fast.py \
  --frames-root ... \
  --output-root ... \
  --start-frame 4308 \
  --end-frame 6171 \
  --index-mode frame
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from tqdm import tqdm
from pathlib import Path


def _scan_camera_records(frame_dir):
    records = []
    for filename in os.listdir(frame_dir):
        if not (filename.endswith(".jpg") and filename.startswith("row_")):
            continue
        try:
            parts = filename[:-4].split("_")
            if len(parts) >= 4 and parts[0] == "row" and parts[2] == "frame":
                csv_row = int(parts[1])
                frame_num = int(parts[3])
                records.append((csv_row, frame_num, filename))
        except (ValueError, IndexError):
            continue
    return records


def _parse_frame_bound(value, bound_name):
    """Parse start/end frame bound; supports int-like values and 'max' for end."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if bound_name == "end" and v == "max":
            return "max"
        try:
            return int(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid {bound_name}_frame={value}. Use integer"
                + (" or 'max'" if bound_name == "end" else "")
            ) from e
    suffix = ', or "max" for end_frame' if bound_name == "end" else ""
    raise ValueError(
        f"Invalid {bound_name}_frame type: {type(value)}. Use integer{suffix}."
    )


def collect_valid_frames(frame_dir, start_frame, end_frame, index_mode="row"):
    """Collect and sort valid frame files by selected index mode."""
    if index_mode not in {"row", "frame"}:
        raise ValueError(f"Invalid index_mode={index_mode}, expected 'row' or 'frame'")
    all_frame_files = []
    for csv_row, frame_num, filename in _scan_camera_records(frame_dir):
        selected_idx = csv_row if index_mode == "row" else frame_num
        all_frame_files.append((selected_idx, csv_row, frame_num, filename))

    if not all_frame_files:
        raise RuntimeError(f"No valid frame images found in {frame_dir}")

    idx_values = [idx for idx, _, _, _ in all_frame_files]
    actual_min_row = min(idx_values)
    actual_max_row = max(idx_values)

    start_frame = _parse_frame_bound(start_frame, "start")
    end_frame = _parse_frame_bound(end_frame, "end")

    adjusted_start = max(start_frame, actual_min_row)
    adjusted_end = actual_max_row if end_frame == "max" else min(end_frame, actual_max_row)
    if adjusted_start > adjusted_end:
        raise RuntimeError(
            f"No frames in requested {index_mode} range {start_frame}~{end_frame}; "
            f"available range is {actual_min_row}~{actual_max_row}"
        )

    valid_frame_files = [
        item
        for item in all_frame_files
        if adjusted_start <= item[0] <= adjusted_end
    ]
    valid_frame_files.sort(key=lambda x: x[0])

    if not valid_frame_files:
        raise RuntimeError(
            f"No valid images after filtering range {adjusted_start}~{adjusted_end}"
        )

    return valid_frame_files, adjusted_start, adjusted_end, actual_min_row, actual_max_row


def collect_valid_frames_by_rows(frame_dir, selected_rows):
    row_to_record = {row: (frame_num, filename) for row, frame_num, filename in _scan_camera_records(frame_dir)}
    valid = []
    missing_rows = []
    for row in selected_rows:
        rec = row_to_record.get(row)
        if rec is None:
            missing_rows.append(row)
            continue
        frame_num, filename = rec
        valid.append((row, row, frame_num, filename))
    if missing_rows:
        raise RuntimeError(
            f"{os.path.basename(frame_dir)} missing {len(missing_rows)} rows, "
            f"first rows: {missing_rows[:10]}"
        )
    return valid


def frames_to_video_opencv(
    valid_frame_files,
    frame_dir,
    output_path,
    fps,
    show_progress=True,
    progress_callback=None,
):
    """Fallback path: encode with OpenCV."""
    first_frame_path = os.path.join(frame_dir, valid_frame_files[0][3])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot create video file: {output_path}")

    try:
        iterator = valid_frame_files
        if show_progress:
            iterator = tqdm(
                valid_frame_files,
                desc=f"OpenCV encode {os.path.basename(output_path)}",
            )
        for _, _, _, filename in iterator:
            frame_path = os.path.join(frame_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            out.write(frame)
            if progress_callback is not None:
                progress_callback(1)
    finally:
        out.release()


def frames_to_video_ffmpeg(
    valid_frame_files,
    frame_dir,
    output_path,
    fps,
    preset,
    crf,
    encoder="x264",
    gpu_id=None,
    show_progress=True,
    progress_callback=None,
):
    """Fast path: encode with ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        for _, _, _, filename in valid_frame_files:
            abs_path = os.path.abspath(os.path.join(frame_dir, filename))
            escaped = abs_path.replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")

    total_frames = len(valid_frame_files)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        "-f",
        "concat",
        "-safe",
        "0",
        "-r",
        str(fps),
        "-i",
        list_path,
    ]
    if encoder == "nvenc":
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                str(crf),
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        )
    try:
        popen_env = os.environ.copy()
        if encoder == "nvenc" and gpu_id is not None:
            popen_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            # Avoid deadlock when ffmpeg writes lots of diagnostics.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=popen_env,
        )
        last_frame = 0
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=total_frames,
                desc=f"ffmpeg encode {os.path.basename(output_path)}",
                unit="frame",
            )

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame="):
                try:
                    cur_frame = int(line.split("=", 1)[1])
                    if cur_frame > last_frame:
                        delta = cur_frame - last_frame
                        if pbar is not None:
                            pbar.update(delta)
                        if progress_callback is not None:
                            progress_callback(delta)
                        last_frame = cur_frame
                except ValueError:
                    pass

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit={ret}) for {output_path}")
        if last_frame < total_frames:
            delta = total_frames - last_frame
            if pbar is not None:
                pbar.update(delta)
            if progress_callback is not None:
                progress_callback(delta)
        if pbar is not None:
            pbar.close()
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def process_single_camera(
    frame_dir,
    output_path,
    start_frame,
    end_frame,
    fps,
    preset,
    crf,
    force_backend,
    show_progress,
    copy_frames_root=None,
    progress_callback=None,
    index_mode="row",
    encoder="x264",
    gpu_id=None,
    valid_frames_override=None,
):
    if valid_frames_override is None:
        valid_frames, adj_start, adj_end, min_row, max_row = collect_valid_frames(
            frame_dir, start_frame, end_frame, index_mode=index_mode
        )
    else:
        valid_frames = valid_frames_override
        idxs = [x[0] for x in valid_frames] if valid_frames else []
        if idxs:
            adj_start, adj_end, min_row, max_row = min(idxs), max(idxs), min(idxs), max(idxs)
        else:
            raise RuntimeError(f"No valid frames selected for {frame_dir}")

    backend = force_backend
    if backend == "auto":
        backend = "ffmpeg" if shutil.which("ffmpeg") else "opencv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if backend == "ffmpeg":
        frames_to_video_ffmpeg(
            valid_frames,
            frame_dir,
            output_path,
            fps,
            preset,
            crf,
            encoder,
            gpu_id,
            show_progress,
            progress_callback,
        )
    elif backend == "opencv":
        frames_to_video_opencv(
            valid_frames, frame_dir, output_path, fps, show_progress, progress_callback
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    copied_frames = 0
    if copy_frames_root is not None:
        camera_name = os.path.basename(frame_dir.rstrip("/"))
        cam_dst_dir = os.path.join(copy_frames_root, camera_name)
        os.makedirs(cam_dst_dir, exist_ok=True)
        for _, _, _, filename in valid_frames:
            shutil.copy2(
                os.path.join(frame_dir, filename),
                os.path.join(cam_dst_dir, filename),
            )
        copied_frames = len(valid_frames)

    return {
        "camera_dir": frame_dir,
        "output_path": output_path,
        "num_frames": len(valid_frames),
        "range": (adj_start, adj_end),
        "available": (min_row, max_row),
        "backend": backend,
        "copied_frames": copied_frames,
    }


def discover_camera_dirs(frames_root):
    camera_dirs = []
    for name in sorted(os.listdir(frames_root)):
        full = os.path.join(frames_root, name)
        if os.path.isdir(full) and name.startswith("Camera"):
            camera_dirs.append((name, full))
    return camera_dirs

def run(
    frames_root,
    output_root,
    output_name,
    start_frame,
    end_frame,
    fps,
    preset,
    crf,
    workers,
    backend,
    copy_frames_root=None,
    index_mode="row",
    encoder="x264",
    gpu_id=None,
    base_camera=None,
):
    camera_dirs = discover_camera_dirs(frames_root)
    if not camera_dirs:
        raise RuntimeError(f"No Camera* folders found under {frames_root}")

    camera_to_dir = {name: d for name, d in camera_dirs}
    selected_by_camera = {}

    if index_mode == "frame" and base_camera is not None:
        if base_camera not in camera_to_dir:
            raise RuntimeError(
                f"Base camera {base_camera} not found. Available: {sorted(camera_to_dir.keys())}"
            )
        base_frames, _, _, _, _ = collect_valid_frames(
            camera_to_dir[base_camera], start_frame, end_frame, index_mode="frame"
        )
        selected_rows = [row for _, row, _, _ in base_frames]
        selected_by_camera[base_camera] = base_frames
        for camera_name, frame_dir in camera_dirs:
            if camera_name == base_camera:
                continue
            selected_by_camera[camera_name] = collect_valid_frames_by_rows(
                frame_dir, selected_rows
            )
    else:
        for camera_name, frame_dir in camera_dirs:
            valid_frames, _, _, _, _ = collect_valid_frames(
                frame_dir, start_frame, end_frame, index_mode=index_mode
            )
            selected_by_camera[camera_name] = valid_frames

    total_frames_all = sum(len(v) for v in selected_by_camera.values())

    jobs = []
    for camera_name, frame_dir in camera_dirs:
        output_path = os.path.join(output_root, camera_name, output_name)
        jobs.append((camera_name, frame_dir, output_path))

    results = []
    errors = []

    show_overall_progress = workers > 1
    overall_pbar = None
    progress_callback = None
    pbar_lock = threading.Lock()
    if show_overall_progress:
        overall_pbar = tqdm(total=total_frames_all, desc="Overall frames", unit="frame")

        def _progress_callback(delta):
            if delta <= 0:
                return
            with pbar_lock:
                overall_pbar.update(delta)

        progress_callback = _progress_callback

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(
                process_single_camera,
                frame_dir,
                output_path,
                start_frame,
                end_frame,
                fps,
                preset,
                crf,
                backend,
                workers == 1,
                copy_frames_root,
                progress_callback,
                index_mode,
                encoder,
                gpu_id,
                selected_by_camera[camera_name],
            ): camera_name
            for camera_name, frame_dir, output_path in jobs
        }
        for fut in as_completed(futures):
            camera_name = futures[fut]
            try:
                result = fut.result()
                results.append((camera_name, result))
            except Exception as e:
                errors.append((camera_name, str(e)))
    if overall_pbar is not None:
        overall_pbar.close()


    for camera_name, msg in sorted(errors, key=lambda x: x[0]):
        print(f"[FAIL] {camera_name}: {msg}")

    if errors:
        raise RuntimeError(f"{len(errors)} camera(s) failed.")


def build_parser():
    parser = argparse.ArgumentParser(description="Fast frame-to-video converter.")
    parser.add_argument("--frames-root", required=True, help="Root containing Camera* dirs")
    parser.add_argument("--output-root", required=True, help="Root output directory")
    parser.add_argument("--output-name", default="sync.mp4", help="Output video filename")
    parser.add_argument(
        "--start-frame",
        required=True,
        help="Start index (inclusive), integer",
    )
    parser.add_argument(
        "--end-frame",
        required=True,
        help="End index (inclusive), integer or 'max'",
    )
    parser.add_argument("--fps", type=int, default=60, help="Output FPS")
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Encoding backend (default: auto)",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="ffmpeg x264 preset (ultrafast, veryfast, medium...)",
    )
    parser.add_argument("--crf", type=int, default=18, help="ffmpeg CRF (lower=better quality)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel cameras")
    parser.add_argument(
        "--encoder",
        choices=["x264", "nvenc"],
        default="x264",
        help="Video encoder for ffmpeg backend",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU id for nvenc (sets CUDA_VISIBLE_DEVICES for ffmpeg process)",
    )
    parser.add_argument(
        "--index-mode",
        choices=["row", "frame"],
        default="row",
        help="Interpret start/end using row_XXXXX or frame_YYYYYYY in filename",
    )
    parser.add_argument(
        "--base-camera",
        default=None,
        help="When index-mode=frame, align other cameras by this camera's selected rows",
    )
    return parser


def main():
    # Old-script style quick config: edit these values and run the file directly.
    # If command-line args are provided, CLI mode takes precedence.
    if len(os.sys.argv) == 1:
        # Defaults are relative so the script is portable across machines.
        script_dir = Path(__file__).resolve().parent
        for VIDEO_NAME in ["1","2","3","4","5"]:
            RAT_NAME = "mice7"
            START_FRAME = 1
            END_FRAME = "max"
            PROJECT_ROOT = str(script_dir)
            FPS = 60


            BACKEND = "auto"  # auto | ffmpeg | opencv
            PRESET = "veryfast"
            CRF = 18
            WORKERS = 3
            INDEX_MODE = "frame"  # "row" or "frame"
            BASE_CAMERA = "Camera1"  # used when INDEX_MODE == "frame"
            ENCODER = "x264"   # "x264" or "nvenc"
            GPU_ID = 3          # Use idle GPU for ffmpeg, or None
            copy_frames_root = None

            if VIDEO_NAME == "sync":
                frames_root = f"{PROJECT_ROOT}/{RAT_NAME}/frames/0.mp4/all"
                output_root = f"{PROJECT_ROOT}/{RAT_NAME}/calibration/videos"
            elif VIDEO_NAME == "cali":
                frames_root = f"{PROJECT_ROOT}/{RAT_NAME}/frames/0.mp4/all"
                output_root = f"{PROJECT_ROOT}/{RAT_NAME}/calibration/videos"
                copy_frames_root = f"{PROJECT_ROOT}/{RAT_NAME}/calibration/frames"
                os.makedirs(copy_frames_root, exist_ok=True)
                for i in range(1, 7):
                    os.makedirs(os.path.join(copy_frames_root, f"Camera{i}"), exist_ok=True)
            else:
                frames_root = f"{PROJECT_ROOT}/{RAT_NAME}/frames/{VIDEO_NAME}.mp4/all"
                output_root = f"{PROJECT_ROOT}/{RAT_NAME}/videos"
            output_name = f"{VIDEO_NAME}.mp4"

            print("===== Fast Frames to Video =====")
            print(f"Frames root:  {frames_root}")
            print(f"Output root:  {output_root}")
            print(f"Output file:  {output_name}")
            print(f"Frame range:  {START_FRAME}~{END_FRAME}")
            print(f"FPS:          {FPS}")
            print(f"Backend:      {BACKEND}")
            print(f"Workers:      {WORKERS}")
            print(f"Index mode:   {INDEX_MODE}")
            print(f"Base camera:  {BASE_CAMERA}")
            print(f"Encoder:      {ENCODER}")
            print(f"GPU ID:       {GPU_ID}")
            print("================================")

            run(
                frames_root=frames_root,
                output_root=output_root,
                output_name=output_name,
                start_frame=START_FRAME,
                end_frame=END_FRAME,
                fps=FPS,
                preset=PRESET,
                crf=CRF,
                workers=WORKERS,
                backend=BACKEND,
                copy_frames_root=copy_frames_root,
                index_mode=INDEX_MODE,
                base_camera=BASE_CAMERA,
                encoder=ENCODER,
                gpu_id=GPU_ID,
            )
        return

    parser = build_parser()
    args = parser.parse_args()
    run(
        frames_root=args.frames_root,
        output_root=args.output_root,
        output_name=args.output_name,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        preset=args.preset,
        crf=args.crf,
        workers=args.workers,
        backend=args.backend,
        copy_frames_root=None,
        index_mode=args.index_mode,
        base_camera=args.base_camera,
        encoder=args.encoder,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
