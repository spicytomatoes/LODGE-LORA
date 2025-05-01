import os
import subprocess
from tqdm import tqdm
import argparse

# --- Configuration ---
DOWNLOAD_PATH = "download"  # Directory to save trimmed videos
# --- End Configuration ---

# Path to log file
LOG_FILE = "process_log.txt"

def log_output(command, stdout, stderr):
    """Log the command output to a log file"""
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Running command: {' '.join(command)}\n")
        log.write("STDOUT:\n")
        log.write(stdout + "\n")
        log.write("STDERR:\n")
        log.write(stderr + "\n")
        log.write("-" * 40 + "\n")

def download_and_trim(url, start_time, end_time, index, ffmpeg_path, download_path):
    """
    Downloads a video using yt-dlp and trims it using ffmpeg based on start/end times.

    Args:
        url (str): The URL of the video to download.
        start_time (int): The start time in seconds for the trim.
        end_time (int): The end time in seconds for the trim.
        index (int): The index used for naming the output file.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Ensure times are integers
    try:
        start_sec = int(start_time)
        end_sec = int(end_time)
        if start_sec < 0 or end_sec <= start_sec:
             print(f"❌ Invalid times for index {index} ({url}): start={start_sec}, end={end_sec}. Skipping.")
             return False
    except ValueError:
        print(f"❌ Invalid time format for index {index} ({url}): start='{start_time}', end='{end_time}'. Skipping.")
        return False


    # --- File Paths ---
    # Temporary filename for the full download
    # Using index ensures uniqueness if the same URL appears multiple times with different trims
    temp_id = f"temp_{index}_{os.path.basename(url).split('=')[-1]}" # Basic attempt at unique temp name
    full_video_path = os.path.join(download_path, f"{temp_id}.mp4")
    # Final filename for the trimmed video
    trimmed_video_path = os.path.join(download_path, f"{index}.mp4")

    try:
        # --- Step 1: Download the full video ---
        yt_dlp_command = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-f", "best[ext=mp4]/best",
            "-o", full_video_path,
            "--no-playlist",
            url
        ]
        yt_dlp_result = subprocess.run(yt_dlp_command, capture_output=True, text=True)
        log_output(yt_dlp_command, yt_dlp_result.stdout, yt_dlp_result.stderr)  # Log yt-dlp output

        # --- Step 2: Trim the video using ffmpeg ---
        ffmpeg_command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", full_video_path,  # Input file
            "-ss", str(start_sec),  # Start time for trimming (after -i for more accuracy)
            "-to", str(end_sec),    # End time for trimming
            "-c:v", "libx264",      # Force re-encoding the video
            "-c:a", "aac",          # Force re-encoding the audio
            "-strict", "experimental",  # Enable experimental codecs if needed
            "-y",                   # Overwrite output file if it exists
            trimmed_video_path      # Output file path
        ]
        ffmpeg_result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        log_output(ffmpeg_command, ffmpeg_result.stdout, ffmpeg_result.stderr)  # Log ffmpeg output

        # --- Step 3: Clean up ---
        os.remove(full_video_path)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed processing index {index} ({url}). Error: {e}")
        # Print stdout/stderr from failed command for debugging
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        # Clean up temp file if it exists
        if os.path.exists(full_video_path):
            try:
                os.remove(full_video_path)
            except OSError as remove_err:
                print(f"Warning: Could not remove temp file {full_video_path}: {remove_err}")
        return False
    except Exception as e: # Catch other potential errors (e.g., file system issues)
        print(f"❌ An unexpected error occurred processing index {index} ({url}): {e}")
        if os.path.exists(full_video_path):
             try:
                os.remove(full_video_path)
             except OSError as remove_err:
                print(f"Warning: Could not remove temp file {full_video_path}: {remove_err}")
        return False


def main(txt_path, ffmpeg_path):
    """
    Main function to read input file, parse tasks, and process videos.
    """
    # --- Ensure download directory exists ---
    download_dir = os.path.join(DOWNLOAD_PATH, os.path.splitext(os.path.basename(txt_path))[0])
    os.makedirs(download_dir, exist_ok=True)

    tasks = []
    print(f"Reading input file: {txt_path}")
    try:
        with open(txt_path, "r", encoding='utf-8') as f: # Added encoding
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue

                parts = line.split(',')
                if len(parts) != 3:
                    print(f"Warning: Skipping invalid line {i+1} (expected 3 parts, got {len(parts)}): {line}")
                    continue

                url = parts[0].strip()
                start_time_str = parts[1].strip()
                end_time_str = parts[2].strip()

                # Basic validation before adding to tasks
                if not url:
                    print(f"Warning: Skipping line {i+1} due to empty URL: {line}")
                    continue
                try:
                    start_time = int(start_time_str)
                    end_time = int(end_time_str)
                    if start_time < 0 or end_time <= start_time:
                         print(f"Warning: Skipping line {i+1} due to invalid time range (start={start_time}, end={end_time}): {line}")
                         continue
                    tasks.append((url, start_time, end_time))
                except ValueError:
                    print(f"Warning: Skipping line {i+1} due to non-integer time values ('{start_time_str}', '{end_time_str}'): {line}")
                    continue

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {txt_path}")
        return
    except Exception as e:
        print(f"❌ Error reading input file {txt_path}: {e}")
        return

    if not tasks:
        print("No valid tasks found in the input file.")
        return

    print(f"Found {len(tasks)} tasks to process.")

    output_index = 0 # Index for the output files (e.g., 0.mp4, 1.mp4, ...)
    success_count = 0
    failure_count = 0

    # Process tasks with a progress bar
    for url, start_time, end_time in tqdm(tasks, desc="Processing videos"):
        success = download_and_trim(url, start_time, end_time, output_index, ffmpeg_path, download_dir)
        if success:
            output_index += 1 # Only increment index if successful
            success_count += 1
        else:
            failure_count += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed:               {failure_count}")
    print(f"Total tasks attempted: {len(tasks)}")
    print(f"Trimmed videos saved in: {os.path.abspath(download_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and trim videos based on a task file.")
    parser.add_argument("txt_path", type=str, help="Path to the input text file (e.g., modern.txt)")
    parser.add_argument("ffmpeg_path", type=str, help="Path to the ffmpeg executable (e.g., C:/ffmpeg/bin/ffmpeg.exe)")
    args = parser.parse_args()

    main(args.txt_path, args.ffmpeg_path)