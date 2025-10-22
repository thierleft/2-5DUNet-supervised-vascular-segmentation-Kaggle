#!/bin/bash
#$ -l tmem=110G
#$ -l h_rt=240:00:00
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 4
#$ -S /bin/bash
#$ -j y
#$ -N train_Heart
#$ -o /home/lefebvre/storage/vasc/logs/

nvidia-smi

echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load conda
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

cd $HOME/storage/vasc/



echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
PORT=$((10000 + RANDOM % 10000))

python /home/lefebvre/storage/vasc/kaggleSegmentation/hra-sennet-hoa-kaggle-2024/winning-team-solutions/team-1/training_code/train.py \
    --port $PORT \
    --memmap_dir /home/lefebvre/storage/vasc/heart/preprocessed_Hs_data \
    --train_groups "LADAF_2021_17|LADAF_2021_17_xz|LADAF_2021_17_zy|LADAF_2021_64|LADAF_2021_64_xz|LADAF_2021_64_zy|LADAF_2024_28|LADAF_2024_28_xz|LADAF_2024_28_zy" \
    --valid_groups "LADAF_2024_56" \
    --epochs 20 \
    --lr 1e-4 \
    --weight_decay 3e-5 \
    --train_batch_size_per_device 6 \
    --valid_batch_size_per_device 6 \
    --accumulation_steps 2 \
    --num_workers 2 \
    --pretrained_weights /home/lefebvre/storage/vasc/training_output_kidney_FINAL_v3/convnext_tiny_2epoch_best.pth \
    --output_dir /home/lefebvre/storage/vasc/heart/newHs1extra_training_output_2prime



echo "End time: $(date)"
