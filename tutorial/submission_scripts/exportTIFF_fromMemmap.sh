#!/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=16G
#$ -l h_rt=10:00:00
#$ -pe smp 4
#$ -S /bin/bash
#$ -j y
#$ -R y 
#$ -N exportTIFFs
#$ -o  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load conda and activate environment
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

# multi-processing flags
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5

# Navigate to working directory
cd $HOME/storage/STORAGESPACE_NAME/

# Run script (NOTE: remove the trailing backslash '\' on the last line or make sure it's multiline)
python exportInference_toTIFF.py \
  --mmap /home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER/Subject01_mask.mmap \
  --shape 1691 1037 785 \
  --out /home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER/TIF/Subject01 \
  --threshold 0.4 --nprocs 48

echo "End time: $(date)"