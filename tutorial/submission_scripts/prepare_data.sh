#!/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -j y
#$ -N prepare_data
#$ -o  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/


echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load conda and activate environment
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

# Navigate to working directory
cd $HOME/storage/STORAGESPACE_NAME/

# Run script (NOTE: remove the trailing backslash '\' on the last line or make sure it's multiline)
python prepare_data.py \
    -s TRAININGDATA_FOLDER \
    -o PREPROCESSEDDATA_FOLDER

echo "End time: $(date)"
