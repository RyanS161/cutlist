#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --job-name=rl_grpo
#SBATCH --output=_jobs/rl_grpo%j.out
#SBATCH --tmp=128G

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model_dir in model_output (e.g. 48059951/checkpoint-22000)>"
  exit 1
fi

FINETUNED_MODEL="$1"


source ~/cutlist/scripts/setup_finetuned_model.sh "$FINETUNED_MODEL"

uv run train_grpo.py \
    --finetuning-data-dir ~/finetuning_data \
    --rl-data-dir ~/rl_tuning_data \
    --base-model-path $TMPDIR/Llama-3.2-1B-Instruct \
    --adapter-path $TMPDIR/finetuned \
    --model-output-dir $SCRATCH/rl_grpo_cutlist/$SLURM_JOB_ID
