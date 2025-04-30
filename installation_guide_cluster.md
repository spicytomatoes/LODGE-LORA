conda env update --file ./lodge.yaml --prune

conda create -n "lodge" python=3.8.20
- login into the cluster with GPU
- enter following command

  ```bash
  eval "$(/home/l/lowsf/miniconda3/bin/conda shell.bash hook)"
  ```

- install conda

  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
  bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh


  ```

- setup environemnt

  ```bash
  conda env create -f ./lodge.yaml
  ```

- install pytorch3d

  ```bash
  pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  ```
