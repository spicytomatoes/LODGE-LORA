# Fine-Tuning Music-to-Dance Generation Using Internet Videos
Most existing music-to-dance generation research relies heavily on the AIST++ dataset, which utilizes a multi-camera setup to capture accurate 3D motion data. However, collecting 3D pose data is both expensive and time-consuming. This project explores the feasibility of fine-tuning state-of-the-art music-to-dance generation models using videos scraped from the Internet, where only 2D or noisy 3D pose estimates are available.

We provide step-by-step instructions on how to train a custom model below, or if you like, how to [use the GUI](#using-the-gui) directly with one of our models.

## 📌 Acknowledgement

This project is built upon the [LODGE (Long Dance Generation)](https://github.com/li-ronghui/LODGE) codebase by Li et al.  
We would like to thank the original authors for their excellent work and open-source contributions.  
If you find this repository useful, please also consider citing the original LODGE paper and acknowledging their repository.


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

### Inference

[Download our LORA models](LINK TO lora_outputs.zip) and put under the root directory.

TODO: Single file inference script?

### Evaluation

[Donwload our datasets](LINK TO PROCESSED DATASET) and put the contents under `/data/`.

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

