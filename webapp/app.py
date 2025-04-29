import os
import sys
import uuid
import threading
import subprocess
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf

# --- Add project root to path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Import project modules ---
try:
    from dld.models.get_model import get_module
    from dld.data.get_data import get_datasets 
    from render import MovieMaker, motion_data_load_process
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure LODGE-LORA modules are accessible and dependencies are installed.")
    sys.exit(1) # Exit if core modules can't be imported

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key' # wont need this for demo
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'webapp', 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- In-memory job tracking ---
jobs = {}

# --- Helper Function for Generation (Runs in Background) ---
def run_generation_and_render(job_id, music_path, output_dir):
    """Runs the core generation and rendering logic."""
    try:
        jobs[job_id]['status'] = 'generating'
        print(f"Job {job_id}: Starting generation...")

        # --- === 1. Load Config (Adapt paths as needed) === ---
        # im basing this on the paths in playground.ipynb
        cfg_local_path = os.path.join(project_root, "exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml")
        cfg_assets_path = os.path.join(project_root, "configs/data/assets.yaml")
        cfg_local = OmegaConf.load(cfg_local_path)
        cfg_assets = OmegaConf.load(cfg_assets_path)
        cfg = OmegaConf.merge(cfg_local, cfg_assets)
        # IMPORTANT: Set correct checkpoint paths relative to project_root
        cfg.checkpoint1 = os.path.join(project_root, 'exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt')
        cfg.checkpoint2 = os.path.join(project_root, 'exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt')
        cfg.TEST.REP_PATH = output_dir # Set output path in config if model uses it
        cfg.DEVICE = '0' # change accordingly

        # --- === 2. Load Model === ---
        # will need to plug and play the finished model here
        # subject to changes
        # dataset = get_datasets(cfg) # May not be needed for generation only
        # model = get_module(cfg, dataset)
        model = get_module(cfg) # Simpler if dataset not needed for get_module
        # Load state dict - adapt from playground.ipynb or your training script
        # state_dict = torch.load(cfg.checkpoint2, map_location="cpu")["state_dict"]
        # model.load_state_dict(state_dict, strict=True)
        # model.eval() # Set to evaluation mode
        # model.to(f"cuda:{cfg.DEVICE}")
        print(f"Job {job_id}: Model loaded (placeholder - adapt loading logic)")

        # --- === 3. Prepare Input (Music Features, etc.) === ---
        # This part is highly dependent on your model's `generate` method
        # Example: If it needs pre-extracted features:
        # music_features = extract_music_features(music_path) # Implement this
        # Example: If it takes the music path directly:
        input_data = {'music_path': music_path, 'genre_id': 0} # Example genre
        print(f"Job {job_id}: Preparing input data...")

        # --- === 4. Run Generation === ---
        # fill in the model generationg code here
        # motion_data = model.generate(**input_data)

        # --- === 5. Render Video === ---
        jobs[job_id]['status'] = 'rendering'
        print(f"Job {job_id}: Starting rendering...")
        # Use render.py 
        render_save_path = os.path.join(output_dir, 'render_output')
        os.makedirs(render_save_path, exist_ok=True)

        # --- Option A: Call render.py as a script ---
        render_script_path = os.path.join(project_root, 'render.py')
        cmd = [
            sys.executable, # Use the current python interpreter
            render_script_path,
            '--modir', output_dir, # Directory containing the .npy file
            '--mode', 'smplh',     # Or smplx, smpl
            '--fps', '30',         # Match your data
            '--save_path', render_save_path,
            '--device', cfg.DEVICE,
            # '--song', music_path # Optional: Add music back
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True) # check=True raises error on failure

        # --- Option B: Use MovieMaker directly (if imports work) ---
        # visualizer = MovieMaker(save_path=render_save_path)
        # Assuming motion_data is already loaded/generated np array
        # processed_motion = motion_data_load_process(motion_npy_path) # If needed
        # visualizer.run(motion_data, tab=job_id, music_file=music_path) # Pass numpy array directly

        # Find the generated video file (adjust name based on render.py output)
        # render.py seems to create files like {tab}-{view_index}-music.mp4 or {tab}-{view_index}.mp4
        # Let's assume it creates job_id-0-music.mp4 or job_id-0.mp4
        video_filename_music = f"{job_id}-0-music.mp4"
        video_filename_no_music = f"{job_id}-0.mp4"
        video_path_music = os.path.join(render_save_path, video_filename_music)
        video_path_no_music = os.path.join(render_save_path, video_filename_no_music)

        final_video_path = None
        if os.path.exists(video_path_music):
             final_video_path = video_path_music
        elif os.path.exists(video_path_no_music):
             final_video_path = video_path_no_music

        if final_video_path:
            # Make path relative to static folder for URL generation
            relative_video_path = os.path.relpath(final_video_path, os.path.join(project_root, 'webapp', 'static'))
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['video_url'] = url_for('static', filename=relative_video_path.replace('\\', '/'))
            print(f"Job {job_id}: Rendering complete. Video URL: {jobs[job_id]['video_url']}")
        else:
             raise FileNotFoundError(f"Generated video not found in {render_save_path}")

    except Exception as e:
        print(f"Job {job_id}: FAILED! Error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_generate():
    """Handle music upload and start generation."""
    if 'music_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['music_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        # Save music file in a job-specific subfolder within uploads
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        music_path = os.path.join(job_dir, filename)
        file.save(music_path)

        jobs[job_id] = {'status': 'queued', 'music_filename': filename}

        # Start background generation
        thread = threading.Thread(target=run_generation_and_render, args=(job_id, music_path, job_dir))
        thread.daemon = True # Allows app to exit even if threads are running
        thread.start()

        flash(f'File uploaded! Generation started (Job ID: {job_id}). Check status below.')
        # Redirect to results page which will poll for status
        return redirect(url_for('results', job_id=job_id))

    return redirect(request.url) # Should not happen if file exists

@app.route('/results/<job_id>')
def results(job_id):
    """Display results page, which will poll for status."""
    if job_id not in jobs:
        flash('Invalid Job ID')
        return redirect(url_for('index'))
    return render_template('results.html', job_id=job_id, job_info=jobs[job_id])

@app.route('/status/<job_id>')
def get_status(job_id):
    """API endpoint to get the status of a job."""
    if job_id not in jobs:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(jobs[job_id])

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Project Root: {project_root}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Static Folder: {os.path.join(app.root_path, 'static')}")
    app.run(debug=True, host='0.0.0.0', port=5000) 