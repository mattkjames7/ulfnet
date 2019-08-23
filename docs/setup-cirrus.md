# Set up Cirrus environment

## Get source code repository

Run:

```bash
git clone https://github.com/marangiop/unet.git
```

This will clone the source code repository into a `unet` directory.

---

## Set up Python environment

First, follow instructions to install miniconda on Cirrus: https://cirrus.readthedocs.io/en/master/user-guide/python.html?highlight=miniconda

Create `unet` environment:

```bash
conda create -n unet

Proceed ([y]/n)? y
```

Activate environment:

```bash
conda activate unet
```
---

## Install dependencies

```bash
cd unet
conda install -c anaconda --file requirements_conda.txt
pip install -r requirements_pip.txt
```

## Build Tensorflow-gpu using Singularity image (required for running on GPUs)

```bash
module load singularity
singularity build tfgpu.simg docker://tensorflow/tensorflow:1.12.0-gpu-py3
``` 

## Install packages inside Tensorflow-gpu image 
```bash
module load singularity
singularity shell tfgpu.simg
conda install -c anaconda --file requirements_conda.txt
pip install -r requirements_pip.txt
```

