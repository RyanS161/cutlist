#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --job-name=rl_grpo
#SBATCH --output=_jobs/rl_grpo%j.out
#SBATCH --tmp=128G

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <job_name> <model_dir in scratch (e.g. rl_grpo/runname/checkpoint-100)>" 
  exit 1
fi


JOB_NAME="$1"
FINETUNED_MODEL="$2"


source ~/cutlist/scripts/setup_finetuned_model.sh "$FINETUNED_MODEL"

uv run train_grpo.py \
    --finetuning-data-dir ~/finetuning_data \
    --rl-data-dir ~/rl_tuning_data \
    --base-model-path $TMPDIR/Llama-3.2-1B-Instruct \
    --adapter-path $TMPDIR/finetuned \
    --model-output-dir $SCRATCH/rl_grpo_cutlist \
    --run-name $JOB_NAME \
