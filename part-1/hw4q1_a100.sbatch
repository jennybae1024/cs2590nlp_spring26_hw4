#!/bin/bash
#SBATCH --job-name=nlp_hw4_q1
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append # Append to the output file. If you don't use this, the output will be overwritten.
#SBATCH --output=./%j_%x.out # %j is the job id, %x is the job name.
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL # Export all the environment variables to the job.
#SBATCH --time=02:00:00 
#SBATCH --gres=gpu:1 # Request 1 GPU.
#SBATCH --requeue # Requeue the job if it fails.

singularity exec --bind /scratch --nv --overlay  /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -lc "
    source ~/.bashrc
    conda activate nlp_hw4
    cd /scratch/$USER/hw4code/release/part-1
    python3 main.py --train --eval
  "

