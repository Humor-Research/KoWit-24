#!/bin/bash
#SBATCH --job-name=wordplay_interpretation_gigachat_lite_extended
#SBATCH --error=wordplay_interpretation_gigachat_lite_extended.err
#SBATCH --output=wordplay_interpretation_gigachat_lite_extended.log
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="type_e"


module purge
module load Python
module load CUDA/12.2
module load cmake/3.21.3
source deactivate
conda activate gigach

srun python3 -u wordplay_interpretation_gigachat_lite_extended.py