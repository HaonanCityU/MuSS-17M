"""
Video Frame Information Extractor

Usage:
    This script processes video files in camera folders and extracts frame information
    (frame number and timestamp) to CSV files.

    The script:
    1. Scans for folders starting with 'Camera' in the specified root directory
    2. Processes all videos ending with the specified video name (e.g., '0.mp4')
    3. Extracts frame number and timestamp (in milliseconds) for each frame
    4. Saves the results to CSV files named: {folder_name}_{video_name}_results.csv
    5. Uses multithreading to process multiple camera folders in parallel

    Example:
        Modify the root_path and video_name in the __main__ section:
        - root_path: Directory containing Camera folders (e.g., Camera1, Camera2, etc.)
        - video_name: Video file suffix to process (e.g., '0' for files ending with '0.mp4')

    Output CSV format:
        frame_number,cap_msec
        0,0
        1,10
        2,26
        ...

Requirements:
    - opencv-python (cv2)
    - tqdm (for progress bars)
"""

from __future__ import annotations

import cv2
import csv
import os
import threading
from tqdm import tqdm
import argparse
from pathlib import Path

def process_camera_folder(folder_path, root_path, video):
    """Process all videos in a single camera folder"""
    folder_name = os.path.basename(folder_path)
    # Store all generated CSVs in a single folder under root_path.
    # Example: .../vessel_1/csv/Camera1_0_results.csv
    csv_dir = os.path.join(root_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    # Iterate through video files in the folder
    for video_file in os.listdir(folder_path):
        if video_file.endswith(f'{video}.mp4'):
            video_path = os.path.join(folder_path, video_file)
            # print(f'Thread {threading.current_thread().name} processing video: {video_path}')
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'Cannot open video: {video_path}')
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            # print(f'Total frames: {frame_count}, FPS: {fps:.2f}')

            # Generate CSV filename
            video_name = os.path.splitext(video_file)[0]
            csv_filename = f"{folder_name}_{video_name}_results.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
    

            # Create progress bar (total frames includes frame 0)
            pbar = tqdm(total=frame_count, desc=f'{folder_name}_{video_name}', unit='frame')

            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['frame_number', 'cap_msec'])
                

                # Step 2: Read video starting from frame 0 and record
                frame_number = 0  # Start counting from 0
                while frame_number < frame_count:  # Total frame count unchanged to avoid writing extra frames
                    # Read current frame (frame frame_number)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Get timestamp of current frame
                    cap_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    csv_writer.writerow([frame_number, round(cap_msec)])

                    pbar.update(1)
                    pbar.set_postfix({
                        'Current frame': frame_number, 
                        'Progress': f'{((frame_number + 1)/frame_count)*100:.2f}%'  # +1 because frame 0 is included
                    })

                    frame_number += 1

            pbar.close()
            cap.release()
            # print(f'CSV file saved to: {csv_path}')

def process_videos(root_path, video_name):
    """Process all camera folders in root directory using multithreading"""
    threads = []
    # Iterate through all subfolders in root folder
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        # Only process folders starting with 'Camera'
        if os.path.isdir(folder_path) and folder_name.startswith('Camera'):
            # Create a thread for each camera folder
            thread = threading.Thread(
                target=process_camera_folder,
                args=(folder_path, root_path, video_name),
                name=folder_name 
            )
            threads.append(thread)
            thread.start()
            print(f'Started thread processing {folder_name}')

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    print(f'All videos in root directory {root_path} processed!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-frame timestamps (cap_msec) for videos in Camera* folders.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing Camera* folders (default: current directory).",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=["0"],
        help="Video suffixes to process (e.g. 0 1 2 for *0.mp4/*1.mp4/*2.mp4).",
    )
    args = parser.parse_args()

    root_path = str(Path(args.root).resolve())
    for video in args.videos:
        process_videos(root_path, video)
    print("Done.")