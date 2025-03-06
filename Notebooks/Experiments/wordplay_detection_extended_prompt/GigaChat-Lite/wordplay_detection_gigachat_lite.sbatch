#!/bin/bash
#SBATCH --job-name=wordplay_detection_gigachat_lite
#SBATCH --error=wordplay_detection_gigachat_lite.err
#SBATCH --output=wordplay_detection_gigachat_lite.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="type_e"


module purge
module load Python
module load CUDA/12.2
module load cmake/3.21.3
source deactivate
conda activate gigach

srun python3 -u wordplay_detection_gigachat_lite.py