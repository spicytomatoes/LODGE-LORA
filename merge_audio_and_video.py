import argparse
import subprocess
import os
import sys

def merge_audio_video(video_path, audio_path, output_path):
    """
    Merges audio from audio_path into video_path and saves to output_path using ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the output video file.
    """
    print(f"Input Video: {video_path}")
    print(f"Input Audio: {audio_path}")
    print(f"Output File: {output_path}")

    # Check if input files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Check if ffmpeg is installed and accessible
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg command not found.")
        print("Please install ffmpeg and ensure it's in your system's PATH.")
        print("On macOS, you can install it using Homebrew: brew install ffmpeg")
        return

    # Construct the ffmpeg command
    # -i video_path : Input video file
    # -i audio_path : Input audio file
    # -c:v copy : Copy video stream without re-encoding (fast)
    # -c:a aac : Encode audio stream to AAC (common codec)
    # -map 0:v:0 : Map video stream from the first input (video_path)
    # -map 1:a:0 : Map audio stream from the second input (audio_path)
    # -shortest : Finish encoding when the shortest input stream ends
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]

    print(f"\nRunning ffmpeg command: {' '.join(cmd)}\n")

    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("ffmpeg stdout:")
        print(result.stdout)
        print("ffmpeg stderr:")
        print(result.stderr)
        print(f"\nSuccessfully merged video and audio into: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution:")
        print(f"Return code: {e.returncode}")
        print("ffmpeg stdout:")
        print(e.stdout)
        print("ffmpeg stderr:")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge an audio file into a video file using ffmpeg.")
    parser.add_argument("-v", "--video", required=True, help="Path to the input video file.")
    parser.add_argument("-a", "--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("-o", "--output", default="output_merged.mp4", help="Path for the output merged video file (default: output_merged.mp4).")

    args = parser.parse_args()

    merge_audio_video(args.video, args.audio, args.output)