#!/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=16G
#$ -l h_rt=10:00:00
#$ -pe smp 4
#$ -S /bin/bash
#$ -j y
#$ -R y 
#$ -N exportmurineK_TIFF
#$ -o /home/lefebvre/storage/vasc/logs/

set -euo pipefail


echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load conda and activate environment
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate vascSeg39

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5

# Navigate to working directory
cd $HOME/storage/vasc/

# Run script (NOTE: remove the trailing backslash '\' on the last line or make sure it's multiline)
python /home/lefebvre/storage/vasc/kaggleSegmentation/hra-sennet-hoa-kaggle-2024/winning-team-solutions/team-1/training_code/exportInference_toTIFF.py \
  --mmap /home/lefebvre/storage/vasc/heart/inference_output_newHs_extra/8_246um_AA11_mouseKidney_overview_mask.mmap \
  --shape 1691 1037 785 \
  --out /home/lefebvre/storage/vasc/heart/inference_output_heart/TIF_Hs_extra1/8_246um_AA11_mouseKidney_overview \
  --threshold 0.4 --nprocs 48

echo "End time: $(date)"