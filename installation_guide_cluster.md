- install conda

  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
  bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
  eval "$(/home/l/lowsf/miniconda3/bin/conda shell.bash hook)"
  ```

- setup environemnt

  ```bash
  conda env create -f ./lodge.yaml
  ```

- install pytorch3d

  ```bash
  pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  ```
