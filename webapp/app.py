import os
import sys
import uuid
import threading
import subprocess
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf

# --- Add project root to path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Define constants before Flask app creation ---
UPLOAD_FOLDER_NAME = 'uploads'
RAW_UPLOADS_SUBDIR = 'raw_uploads' # Subdirectory for original uploads
PREDEFINED_AUDIO_SUBDIR = 'predefined' # Subdirectory for predefined audio

# Calculate absolute paths based on project root and webapp location
WEBAPP_DIR = os.path.join(project_root, 'webapp')
STATIC_DIR = os.path.join(WEBAPP_DIR, 'static')
UPLOAD_FOLDER_ABS = os.path.join(STATIC_DIR, UPLOAD_FOLDER_NAME)
RAW_UPLOADS_DIR_ABS = os.path.join(UPLOAD_FOLDER_ABS, RAW_UPLOADS_SUBDIR)
PRESELECTED_AUDIO_DIR_ABS = os.path.join(STATIC_DIR, PREDEFINED_AUDIO_SUBDIR)


# --- Import project modules ---
try:
    from dld.models.get_model import get_module
    from dld.data.get_data import get_datasets
    # Assuming render.py and its functions are directly importable or in the path
    from render import MovieMaker, motion_data_load_process
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you are running from the 'webapp' directory or the project root is in PYTHONPATH.")
    print("Please ensure LODGE-LORA modules are accessible and dependencies are installed.")
    sys.exit(1) # Exit if core modules can't be imported

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
app.secret_key = 'your_very_secret_key' # Wont need this for demo
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_ABS # Base folder for job outputs AND raw uploads
app.config['RAW_UPLOADS_DIR'] = RAW_UPLOADS_DIR_ABS # Specific folder for raw uploads
app.config['PREDEFINED_FOLDER'] = PRESELECTED_AUDIO_DIR_ABS # Specific folder for predefined
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RAW_UPLOADS_DIR'], exist_ok=True)
os.makedirs(app.config['PREDEFINED_FOLDER'], exist_ok=True)


# --- In-memory job tracking ---
jobs = {}

# --- Helper Function for Generation (Runs in Background) ---
def run_generation_and_render(job_id, music_path_abs, job_output_dir_abs, genre, soft_guidance="True"): # Added soft_guidance
    """Runs the core generation and rendering logic."""
    global jobs # Ensure we modify the global jobs dictionary
    try:
        jobs[job_id]['status'] = 'generating'
        print(f"Job {job_id}: Starting generation via infer_lodge.py...")
        print(f"Job {job_id}: Input Music Path: {music_path_abs}")
        print(f"Job {job_id}: Job Output Directory: {job_output_dir_abs}")
        print(f"Job {job_id}: Genre: {genre}")
        print(f"Job {job_id}: Soft Guidance: {soft_guidance}") # Print soft_guidance


        # --- === 1. Define Paths and Arguments for infer_lodge.py === ---
        infer_script_path = os.path.join(project_root, 'infer_lodge.py')
        cfg_path = "" # Initialize
        cfg_assets_path = "" # Initialize

        # Select config based on genre
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

        # Load main config to potentially get device setting (if needed later)
        try:
            cfg = OmegaConf.load(cfg_path)
            # Add merge from cfg_assets if necessary for device or other params
            # cfg_assets_conf = OmegaConf.load(cfg_assets_path)
            # cfg = OmegaConf.merge(cfg, cfg_assets_conf) # Example merge
        except FileNotFoundError:
             raise FileNotFoundError(f"Config file not found: {cfg_path}")
        except Exception as e:
             raise RuntimeError(f"Error loading OmegaConf config {cfg_path}: {e}")


        device_id = str(cfg.get('DEVICE', '0')) # Get device from config, default to '0'

        # Ensure the script path exists
        if not os.path.exists(infer_script_path):
             raise FileNotFoundError(f"Inference script not found at: {infer_script_path}")

        # --- === 2. Construct Command === ---
        cmd_infer = [
            sys.executable,       
            infer_script_path,
            '--cfg', cfg_path,
            '--cfg_assets', cfg_assets_path,
            '--soft', soft_guidance, # Use the passed soft_guidance value
            '--music_path', music_path_abs, # Absolute path to the music file (in raw_uploads or predefined)
            '--out_dir', job_output_dir_abs,   # Absolute path to the job-specific output directory
            # '--device', device_id,     # Specify the GPU device
        ]
        print(f"Job {job_id}: Running command: {' '.join(cmd_infer)}")

        # --- === 3. Execute infer_lodge.py === ---
        # This will block until infer_lodge.py finishes
        # Run from project root to ensure relative paths in configs work
        result = subprocess.run(cmd_infer, check=True, capture_output=True, text=True, cwd=project_root)
        print(f"Job {job_id}: infer_lodge.py stdout:\n{result.stdout}")
        if result.stderr:
             print(f"Job {job_id}: infer_lodge.py stderr:\n{result.stderr}")
        print(f"Job {job_id}: infer_lodge.py completed successfully.")

        # --- === 4. Locate the generated .npy file (Important!) ---
        # infer_lodge.py likely saves the .npy file inside job_output_dir_abs.
        # We need to find its exact name. Let's assume it's based on the music filename.
        music_basename = os.path.splitext(os.path.basename(music_path_abs))[0]
        # This assumes infer_lodge.py creates output like <music_basename>.npy
        # *** Adjust this logic if infer_lodge.py saves with a different naming convention! ***
        expected_npy_filename = f"{music_basename}.npy"
        motion_npy_path = os.path.join(job_output_dir_abs, expected_npy_filename)

        if not os.path.exists(motion_npy_path):
            # Fallback: try to find *any* .npy file in the output dir if the exact name isn't found
            found_npy = None
            for item in os.listdir(job_output_dir_abs):
                if item.endswith(".npy"):
                    motion_npy_path = os.path.join(job_output_dir_abs, item)
                    print(f"Job {job_id}: Found .npy file: {motion_npy_path}")
                    found_npy = True
                    break
            if not found_npy:
                raise FileNotFoundError(f"Generated motion .npy file not found in {job_output_dir_abs}. Expected pattern '{expected_npy_filename}' or any '.npy'. Check infer_lodge.py output naming.")


        # --- === 5. Render Video === ---
        jobs[job_id]['status'] = 'rendering'
        print(f"Job {job_id}: Starting rendering using motion file: {motion_npy_path}")

        # Define where the render output (video) will be saved
        # Let's put it inside the job_output_dir_abs as well, maybe in a 'render' subdirectory
        render_save_path = os.path.join(job_output_dir_abs, 'render_output')
        os.makedirs(render_save_path, exist_ok=True)

        # --- Option A: Call render.py as a script ---
        render_script_path = os.path.join(project_root, 'render.py')
        if not os.path.exists(render_script_path):
             raise FileNotFoundError(f"Render script not found at: {render_script_path}")

        cmd_render = [
            sys.executable, # Use the current python interpreter
            render_script_path,
            '--modir', job_output_dir_abs, # Directory containing the .npy file (or pass --mofile directly)
            '--mofile', motion_npy_path, # Explicitly pass the motion file path
            '--mode', 'smplh',     # Or smplx, smpl (Make this configurable if needed)
            '--fps', '30',         # Match your data (Make this configurable if needed)
            '--save_path', render_save_path, # Directory to save the video
            '--device', device_id,
            '--tab', job_id,       # Pass job_id to potentially name the output file
            # '--song', music_path_abs # Optional: Add music back DURING rendering
        ]
        print(f"Job {job_id}: Running command: {' '.join(cmd_render)}")
        result_render = subprocess.run(cmd_render, check=True, capture_output=True, text=True, cwd=project_root)
        print(f"Job {job_id}: render.py stdout:\n{result_render.stdout}")
        if result_render.stderr:
             print(f"Job {job_id}: render.py stderr:\n{result_render.stderr}")
        print(f"Job {job_id}: render.py completed successfully.")

        # --- Option B: Use MovieMaker directly (if imports work and preferred) ---
        # print(f"Job {job_id}: Loading motion data from {motion_npy_path}")
        # motion_data = np.load(motion_npy_path)
        # print(f"Job {job_id}: Initializing MovieMaker...")
        # visualizer = MovieMaker(save_path=render_save_path, mode='smplh', fps=30, device=device_id) # Adjust params as needed
        # print(f"Job {job_id}: Running MovieMaker visualization...")
        # visualizer.run(motion_data, tab=job_id, music_file=music_path_abs) # Pass numpy array, job_id for filename, and music
        # print(f"Job {job_id}: MovieMaker run complete.")

        # --- === 6. Find the generated video file === ---
        # Based on render.py or MovieMaker logic, find the output video.
        # Let's assume render.py (or MovieMaker via 'tab' arg) creates a file named like '{job_id}-<something>.mp4'
        # inside 'render_save_path'.
        final_video_path = None
        final_video_relative_path = None
        for filename in os.listdir(render_save_path):
            # Prioritize files containing the job_id and ending with .mp4
            if job_id in filename and filename.lower().endswith(".mp4"):
                 final_video_path = os.path.join(render_save_path, filename)
                 break
        # Fallback if no file with job_id is found (less ideal)
        if not final_video_path:
            for filename in os.listdir(render_save_path):
                 if filename.lower().endswith(".mp4"):
                     final_video_path = os.path.join(render_save_path, filename)
                     print(f"Job {job_id}: Warning - Found video without job_id in name: {filename}")
                     break

        if final_video_path and os.path.exists(final_video_path):
            # Make path relative to the *static* folder for URL generation
            # STATIC_DIR was defined globally earlier
            final_video_relative_path = os.path.relpath(final_video_path, STATIC_DIR)
            # Ensure forward slashes for URL
            final_video_url = url_for('static', filename=final_video_relative_path.replace(os.path.sep, '/'))
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['video_url'] = final_video_url
            print(f"Job {job_id}: Rendering complete. Video Path: {final_video_path}")
            print(f"Job {job_id}: Video URL: {final_video_url}")
        else:
             raise FileNotFoundError(f"Generated video file not found in {render_save_path}. Looked for pattern '{job_id}-*.mp4' or any '*.mp4'.")

    except subprocess.CalledProcessError as e:
        print(f"Job {job_id}: FAILED! Subprocess error during '{' '.join(e.cmd)}'")
        print(f"Job {job_id}: Return Code: {e.returncode}")
        print(f"Job {job_id}: Stdout: {e.stdout}")
        print(f"Job {job_id}: Stderr: {e.stderr}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = f"Script execution failed (see server logs for details). Stderr: {e.stderr[:500]}..." # Limit stderr length
    except FileNotFoundError as e:
        print(f"Job {job_id}: FAILED! File not found error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    except Exception as e:
        print(f"Job {job_id}: FAILED! General error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = f"An unexpected error occurred: {str(e)}"


# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    """Render the main upload page."""
    predefined_files = []
    try:
        # Use PREDEFINED_FOLDER defined earlier
        if os.path.exists(app.config["PREDEFINED_FOLDER"]):
            allowed_extensions = {"wav", "mp3"} # Match index.html accept attribute
            for f in os.listdir(app.config["PREDEFINED_FOLDER"]):
                if '.' in f and f.rsplit('.', 1)[1].lower() in allowed_extensions:
                    # We need the filename relative to the predefined folder for the value
                    predefined_files.append(f)
            predefined_files.sort() # Sort alphabetically
    except Exception as e:
        print(f"Error scanning predefined audio directory {app.config['PREDEFINED_FOLDER']}: {e}")

    return render_template("index.html", predefined_audio_files=predefined_files)


@app.route("/upload", methods=["POST"])
def upload_and_generate():
    """Handle music upload/selection and start generation via AJAX."""
    job_id = str(uuid.uuid4())
    music_path_abs = None
    original_filename = None
    job_output_dir_abs = None # Renamed for clarity

    # --- Get Genre ---
    genre = request.form.get("genre")
    if not genre:
        return jsonify({"error": "Genre was not selected or submitted."}), 400

    # --- Get Soft Guidance (assuming it's a checkbox or similar) ---
    # Default to "True" if not provided or if value is 'on' (typical checkbox value)
    soft_guidance_req = request.form.get("soft_guidance", "true") # Get from form, default 'true'
    soft_guidance = "True" if soft_guidance_req.lower() in ["true", "on", "yes", "1"] else "False"


    # --- Determine Mode and Get Music Path ---
    input_mode = request.form.get("input_mode", "upload")

    # Define the job-specific OUTPUT directory (within base UPLOAD_FOLDER)
    job_output_dir_rel = os.path.join(UPLOAD_FOLDER_NAME, job_id) # Relative to static folder
    job_output_dir_abs = os.path.join(app.config["UPLOAD_FOLDER"], job_id) # Absolute path
    try:
        os.makedirs(job_output_dir_abs, exist_ok=True)
    except OSError as e:
         print(f"Error creating job output directory {job_output_dir_abs}: {e}")
         return jsonify({"error": f"Server error creating job directory: {e}"}), 500

    if input_mode == 'upload':
        if "music_file" not in request.files or not request.files["music_file"].filename:
             return jsonify({"error": "No music file selected for upload."}), 400

        file = request.files["music_file"]
        original_filename = secure_filename(file.filename)

        # Keep backend validation strict if needed, e.g., only WAV for generation
        allowed_extensions = {"wav"}
        if (
            "." not in original_filename
            or original_filename.rsplit(".", 1)[1].lower() not in allowed_extensions
        ):
            return jsonify({
                "error": f'Invalid file type uploaded ({original_filename.rsplit(".", 1)[-1]}). Allowed: {", ".join(allowed_extensions)}'
            }), 400

        # --- Save to the CENTRAL raw uploads directory ---
        # WARNING: This will overwrite files with the same name!
        music_path_abs = os.path.abspath(os.path.join(app.config['RAW_UPLOADS_DIR'], original_filename))
        try:
            file.save(music_path_abs)
            print(f"Upload Mode: Saved '{original_filename}' to central upload dir: {music_path_abs}")
            print(f"Upload Mode: Job output will be in: {job_output_dir_abs}")
        except Exception as e:
             print(f"Error saving file '{original_filename}' to {music_path_abs}: {e}")
             return jsonify({"error": f"Error saving uploaded file: {e}"}), 500

    elif input_mode == 'select':
        selected_filename = request.form.get("selected_audio_path") # This should be just the filename now
        if not selected_filename:
             return jsonify({"error": "No predefined audio file was selected."}), 400

        # Construct the absolute path using the PREDEFINED_FOLDER config
        music_path_abs = os.path.abspath(
            os.path.join(app.config["PREDEFINED_FOLDER"], selected_filename)
        )
        original_filename = selected_filename # Keep track of the base name

        if not os.path.exists(music_path_abs):
            print(f"Error: Predefined file not found at calculated path: {music_path_abs}")
            return jsonify({
                "error": f"Selected predefined file '{selected_filename}' not found on server."
            }), 400

        print(f"Select Mode: Using predefined file '{original_filename}' from {music_path_abs}")
        print(f"Select Mode: Job output will be in: {job_output_dir_abs}")

    else:
        return jsonify({"error": "Invalid input mode specified."}), 400

    # --- Start Background Task ---
    if music_path_abs and job_output_dir_abs and genre:
        jobs[job_id] = {
            "status": "queued",
            "music_filename": original_filename,
            "genre": genre,
            "job_output_dir": job_output_dir_abs, # Store for potential cleanup later
        }
        print(f"Job {job_id} queued. Music: {music_path_abs}, Genre: {genre}, Output Dir: {job_output_dir_abs}")

        thread = threading.Thread(
            target=run_generation_and_render,
            args=(job_id, music_path_abs, job_output_dir_abs, genre, soft_guidance), # Pass soft_guidance
            daemon=True # Use daemon=True
        )
        # thread.daemon = True # Set thread as daemon
        thread.start()

        # Return job_id in JSON on success
        return jsonify({"job_id": job_id}), 202 # 202 Accepted
    else:
        # Should not happen if logic above is correct
        print(f"Error starting task: music_path={music_path_abs}, job_output_dir={job_output_dir_abs}, genre={genre}")
        return jsonify({
            "error": "Failed to prepare generation task. Missing required parameters."
        }), 500


@app.route("/results/<job_id>")
def results(job_id):
    """Display results page, which will poll for status."""
    job_info = jobs.get(job_id)
    if not job_info:
        flash(f"Job ID '{job_id}' not found.")
        return redirect(url_for("index"))
    # Pass basic info needed for the results page display
    return render_template("results.html",
                           job_id=job_id,
                           status=job_info.get('status', 'unknown'),
                           music_filename=job_info.get('music_filename', 'N/A'),
                           genre=job_info.get('genre', 'N/A'))


@app.route("/status/<job_id>")
def get_status(job_id):
    """API endpoint to get the status of a job."""
    job_info = jobs.get(job_id)
    if not job_info:
        return jsonify({"status": "not_found", "error": "Job ID not found."}), 404

    response = {
        'status': job_info.get('status', 'unknown'),
        'music_filename': job_info.get('music_filename', 'N/A'), # Also return filename/genre if needed
        'genre': job_info.get('genre', 'N/A')
    }
    if job_info.get('status') == 'completed':
        response['video_url'] = job_info.get('video_url')
    elif job_info.get('status') == 'failed':
        response['error'] = job_info.get('error', 'An unknown error occurred.')

    return jsonify(response)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Flask Application...")
    print(f"Project Root: {project_root}")
    print(f"Webapp Directory: {WEBAPP_DIR}")
    print(f"Static Directory: {STATIC_DIR}")
    print(f"Base Upload/Output Folder (Absolute): {app.config['UPLOAD_FOLDER']}")
    print(f"Raw Uploads Folder (Absolute): {app.config['RAW_UPLOADS_DIR']}")
    print(f"Predefined Audio Folder (Absolute): {app.config['PREDEFINED_FOLDER']}")
    print(f"Static Folder served by Flask: {app.static_folder}") # Check Flask's understanding

    # Note: Running with debug=True and host='0.0.0.0' is convenient for development
    # but consider security implications for production. Debug mode exposes vulnerabilities.
    app.run(debug=True, host="0.0.0.0", port=5000)