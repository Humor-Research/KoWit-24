#!/bin/bash
#SBATCH --job-name=wordplay_interpretation_mistral_nemo
#SBATCH --error=wordplay_interpretation_mistral_nemo.err
#SBATCH --output=wordplay_interpretation_mistral_nemo.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="type_e"


module purge
module load Python
module load CUDA/11.7
module load cmake/3.21.3
source deactivate
conda activate mist_v

srun python3 -u wordplay_interpretation_mistral_nemo.py