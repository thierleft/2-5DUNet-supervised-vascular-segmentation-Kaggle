#!/bin/bash
#$ -l tmem=110G
#$ -l h_rt=52:00:00
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 2
#$ -S /bin/bash
#$ -j y
#$ -N infer
#$ -o  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/

echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load cuda/conda/env
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

nvidia-smi 

# Move to working directory
cd $HOME/storage/STORAGESPACE_NAME/


SEED=42
BACKBONE=convnext_tiny
CKPT_PATH="PRETRAINEDWEIGHTS_FOLDER/PRETRAINEDWEIGHTS.pth"
IN_CHANNELS=3
NUM_CLASSES=3
BATCH_SIZE=6
AXIS="z|y|x"
FLIP=3
ROT=3


BASE_DIR="/home/ID/storage/STORAGESPACE_NAME/INFERENCEDATA_FOLDER" 
OUT_DIR="/home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER" 

# List of groups (subfolders with TIFFs)
groups=(
    "Subject01"
    "Subject02"
    "Subject03"
    "Subject04"
)

for group in "${groups[@]}"; do
    RAW_PATH="${BASE_DIR}/${group}"

    echo "Running inference on ${RAW_PATH}"

    python inference.py \
        --seed $SEED \
        --group $group \
        --backbone $BACKBONE \
        --ckpt_path $CKPT_PATH \
        --in_channels $IN_CHANNELS \
        --num_classes $NUM_CLASSES \
        --batch_size $BATCH_SIZE \
        --axis $AXIS \
        --flip $FLIP \
        --rot $ROT \
        --overlap \
        --input_folder $RAW_PATH \
        --output_folder $OUT_DIR
    
    echo "Finished inference for ${group}"
done

echo "End time: $(date)"
