import subprocess
import os
import sys
from argparse import ArgumentParser

project_root = "."

def generate_render_combine(music_path_abs, genre,):
    if genre == "kpop":
        cfg_path = os.path.join(project_root, 'configs/lodge/lora_local_kpop.yaml')
        cfg_assets_path = os.path.join(project_root, 'configs/data/assets-kpop.yaml')
    elif genre == "chinese":
        cfg_path = os.path.join(project_root, 'configs/lodge/lora_local_chinese.yaml')
        cfg_assets_path = os.path.join(project_root, 'configs/data/assets-chinese.yaml')
    elif genre == "ballet":
        cfg_path = os.path.join(project_root, 'configs/lodge/lora_local_ballet.yaml')
        cfg_assets_path = os.path.join(project_root, 'configs/data/assets-ballet.yaml')
    elif genre == "modern":
        cfg_path = os.path.join(project_root, 'configs/lodge/lora_local_modern.yaml')
        cfg_assets_path = os.path.join(project_root, 'configs/data/assets-modern.yaml')
    else:
            raise ValueError(f"Invalid genre '{genre}' provided for config selection.")

    infer_script_path = "./infer_lodge.py"
    cmd_infer = [
        "python",      
        infer_script_path,
        '--cfg', cfg_path,
        '--cfg_assets', cfg_assets_path,
        '--music_path', music_path_abs,
    ]
    try:
        result_infer = subprocess.run(cmd_infer, check=True, capture_output=True)
        print("Inference Stdout:\n", result_infer.stdout)
        if result_infer.stderr:
            print("Inference Stderr:\n", result_infer.stderr)
    except subprocess.CalledProcessError as e:
        print(f"!!! Inference FAILED (Exit Code: {e.returncode}) !!!")
        print(f"Command: {' '.join(e.cmd)}")
        print("--- Inference Stdout ---")
        print(e.stdout)
        print("--- Inference Stderr (Error Output) ---")
        print(e.stderr) 
        print("-----------------------")
        raise

    FILENAME = music_path_abs[:-4]
    INPUT_AUDIO_PATH =  os.path.join("./demo/audio", music_path_abs)

    render_cmd = [
        "python", "./render_single.py",
        "--mofile", f"experiments/concat/npy/{FILENAME}.npy",
        "--save_path", "demo/rendered"
    ]

    # Step 2: Run main2.py
    merge_cmd = [
        "python", "./merge_audio_and_video.py",
        "--video", f"demo/rendered/{FILENAME}.mp4",
        "--audio", INPUT_AUDIO_PATH,
        "--output", f"demo/{FILENAME}_video.mp4"
    ]

    # Execute the commands
    subprocess.run(render_cmd, check=True)
    subprocess.run(merge_cmd, check=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    group = parser.add_argument_group("script params")
    group.add_argument(
        "--filename",
        type=str,
        required=True,
        help="name of file",
    )
    group.add_argument(
        "--genre",
        type=str,
        required=True,
        help="genre",
    )
    params = parser.parse_args()
    generate_render_combine(params.filename, params.genre)

