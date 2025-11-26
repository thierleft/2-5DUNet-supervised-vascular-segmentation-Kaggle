#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=6:00:00
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -S /bin/bash
#$ -j y
#$ -N envInstall
#$ -o /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/


echo "Running on host: $(hostname)"
echo "Start time: $(date)"

cd $HOME/storage/vasc/
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load conda
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/anaconda/conda-2022-5.source

# this environment should be created and then moved to your local project storage space before installing additional packages like here
# conda create -n vascSeg39 python=3.9
conda activate vascSeg39

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install numpy==1.26.4 albumentations==1.3.1 colorama==0.4.6 efficientnet_pytorch==0.7.1 einops==0.7.0 Geometry3D==0.2.4 loguru==0.7.2 matplotlib==3.8.2 numba==0.58.1 \
    numpy==1.26.4 opencv_python==4.8.1.78 pandas==2.2.0 Pillow==10.2.0 pretrainedmodels==0.7.4 scipy==1.12.0 timm==0.9.10 tqdm==4.66.1 transformers==4.35.2

# Validate
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"

echo "End time: $(date)"
