#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name=finetunedata
#SBATCH --output=_jobs/finetunedata_%j.out
#SBATCH -c 12
#SBATCH --tmp=256G

echo "Starting job -- $(date)"

echo "Copying partnet data from scratch -- $(date)"
cp $SCRATCH/data/cleaned_data_v0.tar $TMPDIR/

echo "Copying brickgpt data from scratch -- $(date)"
cp $SCRATCH/data/stabletext2brick.tar $TMPDIR/


cd $TMPDIR/

echo "Extracting partnet data -- $(date)"
tar -xf cleaned_data_v0.tar

echo "Extracting brickgpt data-- $(date)"
tar -xf stabletext2brick.tar



cd ~/cutlist/
source ./scripts/eulerScripts/setup_euler_python.sh 


echo "Running generation script -- $(date)"

uv run ./generate_data.py \
    --partnet_dir $TMPDIR/data_v0 \
    --brickgpt_dir $TMPDIR/StableText2Brick/data \
    --output $TMPDIR/

echo "Copying data back -- $(date)"
cp -r $TMPDIR/finetuning_data $SCRATCH/data/

echo "Job completed -- $(date)"
