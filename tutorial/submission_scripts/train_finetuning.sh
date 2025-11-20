#!/bin/bash
#$ -l tmem=110G
#$ -l h_rt=240:00:00
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 4
#$ -S /bin/bash
#$ -j y
#$ -N train_finetuning
#$ -o /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/

nvidia-smi

echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load cuda/conda/env
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

cd $HOME/storage/STORAGESPACE_NAME/


echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
PORT=$((10000 + RANDOM % 10000))

python train.py \
    --port $PORT \
    --memmap_dir PREPROCESSEDDATA_FOLDER \
    --train_groups "Subject01|Subject01_xz|Subject01_zy|Subject02|Subject02_xz|Subject02_zy" \
    --valid_groups "SubjectN|SubjectN_xz|SubjectN_zy" \
    --epochs 20 \
    --lr 1e-4 \
    --weight_decay 3e-5 \
    --train_batch_size_per_device 6 \
    --valid_batch_size_per_device 6 \
    --accumulation_steps 2 \
    --num_workers 2 \
    --pretrained_weights PRETRAINEDWEIGHTS_FOLDER/PRETRAINEDWEIGHTS.pth \
    --output_dir OUTPUT_FOLDER



echo "End time: $(date)"
