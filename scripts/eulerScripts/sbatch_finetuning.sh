#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --job-name=finetuning
#SBATCH --output=_jobs/finetuning%j.out
#SBATCH --tmp=128G

MODEL_NAME="Llama-3.2-1B-Instruct"


echo "Copying model and extracting model from scratch -- $(date)"

cp -r $SCRATCH/data/finetuning_data $TMPDIR/
cp $SCRATCH/$MODEL_NAME.tar $TMPDIR

tar -xf $TMPDIR/$MODEL_NAME.tar -C $TMPDIR/

echo "Starting Training -- $(date)"

cd ~/cutlist

source ./scripts/eulerScripts/setup_euler_python.sh

source ./scripts/finetune.sh $TMPDIR/$MODEL_NAME $SCRATCH/model_output/ $SLURM_JOB_ID $TMPDIR/finetuning_data
