#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH --job-name=finetuning
#SBATCH --output=_jobs/finetuning%j.out
#SBATCH --tmp=32G



cd ~/cutlist
source ./scripts/eulerScripts/setup_euler_python.sh

source ./scripts/finetune.sh ~/Llama-3.2-1B-Instruct ~/model_output $SLURM_JOB_ID ~/finetuning_data
