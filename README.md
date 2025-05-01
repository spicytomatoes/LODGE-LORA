# Fine-Tuning Music-to-Dance Generation Using Internet Videos
Most existing music-to-dance generation research relies heavily on the AIST++ dataset, which utilizes a multi-camera setup to capture accurate 3D motion data. However, collecting 3D pose data is both expensive and time-consuming. This project explores the feasibility of fine-tuning state-of-the-art music-to-dance generation models using videos scraped from the Internet, where only 2D or noisy 3D pose estimates are available.

We provide step-by-step instructions on how to train a custom model below, or if you like, how to [use the GUI](#using-the-gui) directly with one of our models.

## Installation

The code is tested to run on linux with python 3.8, CUDA 12.6.

Install required libraries:
```bash
apt-get update
apt-get install libsndfile1
apt-get install libosmesa6-dev
apt install freeglut3-dev
```

Create conda environment:
```bash
conda create -n lodge-lora python=3.8
pip install -r requirements.txt
```

To run the fine-tuning script, you also need to download the pretrained models and SMPLX models.

[Download LODGE Pre-trained Models here.](https://drive.google.com/file/d/13Yp__EPAw0EjrSS898X5FtSQGmveBykA/view?usp=sharing)

## Training your own model
To train a custom model, follow the steps below.

### Collecting videos
Collate a *.txt* file with each line containing the following:
>{video_url}, {start_time_in_seconds}, {end_time_in_seconds}

Note: For best performance, select videos with:
* a stationary camera view
* only one person
* full body always seen
* not too baggy clothing
* no holding of props

Install [FFmpeg](https://www.ffmpeg.org/) and run this command:
> python download_videos.py -txt_path [path to *.txt* file] -ffmpeg_path [path to ffmpeg folder]

This will create a folder with the same name as the *.txt* file containing all the downloaded videos.

### Extracting poses from videos

Since the dataset format follows the [FineDance](https://github.com/li-ronghui/FineDance) dataset format, it consists of the following directories:

```bash
├──label_json   # Contains the metadata of each sample in the dataset
├──motion       # Contains the motion data
├──music_npy    # Contains music features
├──music_wav    # Contains music audio
```

The `music_wav` and `music_npy` files can be easily generated from a video using `ffmpeg` and `librosa`, respectively. However, generating the motion data requires using an SMPL-X pose estimation model to output the necessary parameters. We decided to use [SMPLest-X](https://github.com/SMPLCap/SMPLest-X), as it is a state-of-the-art model for estimating 3D human pose in SMPL-X format.


For dataset generation, we used `ffmpeg` to convert videos into a sequence of images. These images were then passed through the YOLOv8 model to detect humans in each frame. The detected human images were fed into the SMPLest-X model to output a set of motion parameters required for the `motion` data.

The outputs from SMPLest-X — including `cam_trans`, `root_pose`, `global_orient`, `lhand_pose`, and `rhand_pose` — were converted into `rot6d` format to match the `FineDance` dataset specification.

However, we encountered some issues during data preprocessing:
- Converting `cam_trans` to `root_trans` was unclear due to missing camera information.
- Some frames were missing due to YOLOv8 failing to detect humans in certain images.


To address these problems, we explored several approaches. One attempt involved using another pose estimation model, [MediaPipe](https://github.com/google-ai-edge/mediapipe), and estimating the x, y, and z translation parameters using the relative position of the pelvis and the lowest point of the estimated pose. However, this approach produced unstable results. Ultimately, we opted to manually tune the camera parameters (such as focal length and principal point) in the configuration file to estimate the `root_trans` values.

For the missing frames, we applied interpolation methods to fill in the gaps. Simply removing the frames would have caused misalignment between the music and motion, resulting in an unsynchronized and lower-quality dataset.


### Preprocess Pose
Run preprocessing script on your dataset.
```bash
python data/code/preprocess.py --data_dir data/your_dataset_name
python dld/data/pre/FineDance_normalizer.py --dataset your_dataset_name
```

Your dataset should have file structure like below:
```bash
LODGE
├── data
│   ├── code
│   │   ├──preprocess.py
│   │   ├──extract_musicfea35.py
│   ├── your_dataset_name
│   │   ├──label_json
│   │   ├──motion
│   │   ├──music_npy
│   │   ├──music_wav
│   │   ├──music_npynew
│   │   ├──mofea319
│   │── Normalizer.pth
└   └── smplx_neu_J_1.npy
```

### Fine-tuning the model
Prepare the asset and training configs for your dataset. You can see the example configs under `configs/data` and `configs/lodge`. Then run the training script

```bash
python train.py --cfg configs/lodge/your_training_config.yaml --cfg_assets configs/data/your_asset_config.yaml
```

The folder structure should be:
```bash
LODGE
├── lora_outputs
│   ├── ballet
│   ├── chinese
│   ├── kpop
│   ├── modern
...
```

### Inference

[Download our LORA models](https://nusu-my.sharepoint.com/:u:/g/personal/e1374097_u_nus_edu/Eac63-6wdFVIkRRmfv6Iu9gBDDzJARl9Md_o-nxXiRyaQA?e=F3LkWz) and put under the root directory.

TODO: Single file inference script?

### Evaluation

[Donwload our datasets](https://nusu-my.sharepoint.com/:u:/g/personal/e1374097_u_nus_edu/EZi-ZJ8WCpJLgdU0SIPwywwBvjvhF78ycK4enyOOts8mOQ?e=Yxldog) and put the contents under `/data/`.

The folder structure should be:
```bash
LODGE
├── data
│   ├── finedance-ballet
│   ├── finedance-chinese
│   ├── finedance-kpop
│   ├── finedance-modern
│   ├── ...
...
```

```bash
# Ballet
python infer_eval.py --cfg configs/lodge/lora_local_ballet.yaml --cfg_assets configs/data/assets-ballet.yaml --soft 1.0 --exp_dir lora_outputs/ballet --name LoRA_Ballet
# Chinese
python infer_eval.py --cfg configs/lodge/lora_local_chinese.yaml --cfg_assets configs/data/assets-chinese.yaml --soft 1.0 --exp_dir lora_outputs/chinese --name LoRA_Chinese
# K-Pop
python infer_eval.py --cfg configs/lodge/lora_local_kpop.yaml --cfg_assets configs/data/assets-kpop.yaml --soft 1.0 --exp_dir lora_outputs/kpop --name LoRA_Kpop
# Modern
python infer_eval.py --cfg configs/lodge/lora_local_modern.yaml --cfg_assets configs/data/assets-modern.yaml --soft 1.0 --exp_dir lora_outputs/modern --name LoRA_Modern
```

## Using the GUI

