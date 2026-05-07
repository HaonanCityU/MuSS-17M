"""
Video File Renamer

Usage:
    This script renames all video files in Camera folders (Camera1-Camera6) to sequential 
    numeric names starting from 0 (0.mp4, 1.mp4, 2.mp4, ...).

    The script:
    1. Scans the parent folder for Camera1-Camera6 folders
    2. For each Camera folder, finds all video files (supports .mp4, .avi, .mov, .mkv, etc.)
    3. Sorts files by their original names (natural sort order)
    4. Renames them sequentially starting from 0
    5. Preserves the original file extension
    6. Prints the mapping of old names to new names for each Camera folder

    Example:
        python 0_rename_folder.py
        
        Or modify the parent_folder_path variable in the __main__ section:
        parent_folder_path = "path/to/parent/folder"  # Should contain Camera1, Camera2, etc.

    Supported video formats:
        .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v

    Note:
        - Files are renamed in place (original names are lost)
        - Make sure to backup your files before running this script
        - The script preserves the original file extension
        - Only processes Camera1, Camera2, Camera3, Camera4, Camera5, Camera6 folders
"""

import os
import re
from pathlib import Path
import shutil
import argparse

def natural_sort_key(text):
    """
    Generate a key for natural sorting (handles numbers correctly)
    Example: ['file1.mp4', 'file10.mp4', 'file2.mp4'] -> ['file1.mp4', 'file2.mp4', 'file10.mp4']
    """
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split(r'(\d+)', text)]

def get_video_files(folder_path):
    """
    Get all video files from the specified folder, sorted naturally
    
    Args:
        folder_path: Path to the folder containing video files
        
    Returns:
        List of video file paths sorted by natural order
    """
    # Supported video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    
    video_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if it's a file and has a video extension
        if os.path.isfile(file_path):
            file_ext = Path(file_name).suffix.lower()
            if file_ext in video_extensions:
                video_files.append(file_path)
    
    # Sort files using natural sort (handles numbers correctly)
    video_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return video_files


def get_video_files_recursive(folder_path):
    """
    Recursively collect video files under folder_path and sort naturally.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_ext = Path(file_name).suffix.lower()
            if file_ext in video_extensions:
                video_files.append(os.path.join(root, file_name))
    video_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return video_files


def move_local_videos_to_camera_root(camera_folder_path, local_folder_name="local"):
    """
    If CameraX/local/ exists, move all videos under it into CameraX/ root,
    then delete the local folder.
    """
    local_name_lower = local_folder_name.lower()

    # Case-insensitive match: e.g. 'local', 'Local', 'LOCAL'
    candidate_local_dirs = []
    for d in os.listdir(camera_folder_path):
        full = os.path.join(camera_folder_path, d)
        if os.path.isdir(full) and d.lower() == local_name_lower:
            candidate_local_dirs.append(full)

    if not candidate_local_dirs:
        return

    for local_path in candidate_local_dirs:
        video_files = get_video_files_recursive(local_path)
        if not video_files:
            # Safety: if we didn't find any recognizable video files, do NOT delete local/
            # (avoid accidental deletion if extensions differ or files are not accessible).
            print(
                f"Warning: No video files found under '{local_path}'. "
                f"Skipping deletion to avoid data loss."
            )
            continue

        # Move videos into the camera root while avoiding name collisions.
        moved_count = 0
        failed = []
        for idx, src_path in enumerate(video_files):
            base_name = os.path.basename(src_path)
            dst_path = os.path.join(camera_folder_path, base_name)
            if os.path.exists(dst_path):
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                dst_path = os.path.join(
                    camera_folder_path, f"{stem}__local__{idx}{suffix}"
                )

            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                failed.append((src_path, dst_path, str(e)))
                print(f"Error moving '{src_path}' -> '{dst_path}': {e}")

        # If any move failed, do NOT delete local/ to avoid data loss.
        if failed:
            print(
                f"Aborting deletion: {len(failed)} file(s) failed to move from "
                f"'{local_path}' to '{camera_folder_path}'."
            )
            continue

        # Delete the local folder after moving all videos.
        shutil.rmtree(local_path, ignore_errors=True)
        print(
            f"Moved {moved_count} video(s) from '{os.path.basename(local_path)}' into "
            f"'{camera_folder_path}' and removed '{os.path.basename(local_path)}'."
        )

def rename_videos(folder_path):
    """
    Rename all video files in the folder to sequential numeric names
    
    Args:
        folder_path: Path to the folder containing video files
        
    Returns:
        List of tuples: (old_name, new_name)
    """
    # Get all video files sorted by natural order
    video_files = get_video_files(folder_path)
    
    if not video_files:
        print(f"No video files found in: {folder_path}")
        return []
    
    print(f"\nFound {len(video_files)} video file(s) in: {folder_path}\n")
    print("=" * 80)
    print(f"{'Original Name':<50} {'New Name':<30}")
    print("=" * 80)
    
    rename_mapping = []
    
    # First pass: rename to temporary names to avoid conflicts
    temp_files = []
    for idx, old_path in enumerate(video_files):
        old_name = os.path.basename(old_path)
        file_ext = Path(old_name).suffix
        temp_name = f"__temp_{idx}{file_ext}"
        temp_path = os.path.join(folder_path, temp_name)
        
        os.rename(old_path, temp_path)
        temp_files.append((temp_path, old_name, file_ext))
    
    # Second pass: rename from temporary names to final names
    for idx, (temp_path, old_name, file_ext) in enumerate(temp_files):
        new_name = f"{idx}{file_ext}"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(temp_path, new_path)
        rename_mapping.append((old_name, new_name))
        
        # Print the mapping
        print(f"{old_name:<50} {new_name:<30}")
    
    print("=" * 80)
    print(f"\nSuccessfully renamed {len(rename_mapping)} file(s).\n")
    
    return rename_mapping

def find_camera_folders(parent_folder_path):
    """
    Find all Camera1-Camera6 folders in the parent folder
    
    Args:
        parent_folder_path: Path to the parent folder containing Camera folders
        
    Returns:
        List of Camera folder paths (Camera1, Camera2, ..., Camera6)
    """
    camera_folders = []
    
    if not os.path.exists(parent_folder_path):
        print(f"Error: Parent folder does not exist: {parent_folder_path}")
        return camera_folders
    
    if not os.path.isdir(parent_folder_path):
        print(f"Error: Path is not a directory: {parent_folder_path}")
        return camera_folders
    
    # Look for Camera1 through Camera6 folders
    for i in range(1, 7):
        camera_folder_name = f"Camera{i}"
        camera_folder_path = os.path.join(parent_folder_path, camera_folder_name)
        
        if os.path.exists(camera_folder_path) and os.path.isdir(camera_folder_path):
            camera_folders.append(camera_folder_path)
        else:
            print(f"Warning: {camera_folder_name} folder not found in: {parent_folder_path}")
    
    return camera_folders

def process_all_camera_folders(parent_folder_path):
    """
    Process all Camera1-Camera6 folders and rename videos in each
    
    Args:
        parent_folder_path: Path to the parent folder containing Camera folders
    """
    camera_folders = find_camera_folders(parent_folder_path)
    
    if not camera_folders:
        print(f"\nNo Camera folders (Camera1-Camera6) found in: {parent_folder_path}")
        return
    
    print(f"Processing {len(camera_folders)} Camera folder(s) in: {parent_folder_path}")
    
    # Process each Camera folder
    for camera_folder in camera_folders:
        camera_name = os.path.basename(camera_folder)
        print(f"Processing: {camera_name}")
        move_local_videos_to_camera_root(camera_folder)
        rename_videos(camera_folder)
    
    print(f"All Camera folders processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename videos under Camera1..Camera6 to 0.mp4, 1.mp4, ...",
    )
    parser.add_argument(
        "parent_folder",
        nargs="?",
        default=".",
        help="Folder containing Camera1..Camera6 (default: current directory).",
    )
    args = parser.parse_args()

    parent_folder_path = str(Path(args.parent_folder).resolve())
    process_all_camera_folders(parent_folder_path)
