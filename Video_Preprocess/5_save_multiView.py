"""
Multi-Camera Video Combiner

Usage:
    This script combines multiple synchronized camera videos into a single multi-view 
    video with a 3x2 grid layout. Each camera view is displayed in a separate grid cell 
    with a camera label overlay.

    The script:
    1. Opens up to 6 video files (Camera1 through Camera6)
    2. Reads frames from all videos simultaneously
    3. Resizes all frames to the same size (using the minimum dimensions)
    4. Arranges frames in a 3x2 grid layout:
       - Row 1: Camera1, Camera2, Camera3
       - Row 2: Camera4, Camera5, Camera6
    5. Adds camera labels to each frame
    6. Combines frames and saves as a single output video

    Layout:
        +-----------+-----------+-----------+
        | Camera1   | Camera2   | Camera3   |
        +-----------+-----------+-----------+
        | Camera4   | Camera5   | Camera6   |
        +-----------+-----------+-----------+

    Input Requirements:
        - Up to 6 video files (MP4 format recommended)
        - Videos should be synchronized (same frame count and FPS)
        - If fewer than 6 videos are provided, empty black frames will be used

    Output:
        - Single combined video file (MP4 format)
        - Frame rate matches the first input video
        - Resolution: (min_width * 3) x (min_height * 2)

    Example:
        Modify the parameters in the main() function:
        - base_dir: Base directory containing Camera folders
        - video_names: List of video file paths relative to base_dir
        - output_path: Output video file path

    Requirements:
        - opencv-python (cv2)
        - numpy
"""

import cv2
import os
import numpy as np
import argparse
from pathlib import Path

def save_multi_view(video_paths, output_path):
    """
    Combine multiple videos into a multi-view video and save
    
    Args:
        video_paths: List containing up to 6 video file paths
        output_path: Output video file path
    """
    # Open all videos
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {path}")
            return None
        caps.append(cap)
    
    # Get video information
    fps = caps[0].get(cv2.CAP_PROP_FPS) if caps else 30
    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT)) if caps else 0
    
    print(f"Starting video processing, total {frame_count} frames, FPS: {fps}")
    
    # Read first frame to determine dimensions
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    
    # Ensure there are 6 frames
    while len(frames) < 6:
        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    # Resize all frames to the same size (using minimum dimensions)
    min_height = min([frame.shape[0] for frame in frames])
    min_width = min([frame.shape[1] for frame in frames])
    
    # Calculate output video dimensions (3x2 layout)
    output_width = min_width * 3
    output_height = min_height * 2
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file {output_path}")
        for cap in caps:
            cap.release()
        return None
    
    # Reset all videos to start position
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_num = 0
    while True:
        # Read current frame from all videos
        frames = []
        all_finished = True
        
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                all_finished = False
            else:
                # If a video has ended, use black frame
                frames.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
        
        # If all videos have ended, exit loop
        if all_finished:
            break
        
        # Ensure there are 6 frames
        while len(frames) < 6:
            frames.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
        
        # Resize all frames to the same size
        resized_frames = []
        for i, frame in enumerate(frames):
            # Resize frame
            resized = cv2.resize(frame, (min_width, min_height))
            # Add camera label
            cv2.putText(resized, f"Camera{i+1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            resized_frames.append(resized)
        
        # Create 3x2 layout
        # Row 1: Camera1, Camera2, Camera3
        row1 = np.hstack([resized_frames[0], resized_frames[1], resized_frames[2]])
        # Row 2: Camera4, Camera5, Camera6
        row2 = np.hstack([resized_frames[3], resized_frames[4], resized_frames[5]])
        # Combine two rows
        combined = np.vstack([row1, row2])
        
        # Write to video
        out.write(combined)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames...")
    
    # Release all resources
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"Video saved successfully! Processed {frame_num} frames, saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine up to 6 camera videos into a 3x2 grid montage.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directory containing Camera1..Camera6 folders (default: current directory).",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="0.mp4",
        help="Video filename under each CameraX folder (default: 0.mp4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multicamera_combined.mp4",
        help="Output MP4 path (default: base-dir/multicamera_combined.mp4).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    video_paths = [str(base_dir / f"Camera{i}" / args.video_name) for i in range(1, 7)]
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Warning: video file does not exist: {path}")

    output_path = str((base_dir / args.output).resolve())
    save_multi_view(video_paths, output_path)

if __name__ == "__main__":
    main()
