if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <model_dir in scratch (e.g. model_output/48059951/checkpoint-22000)> "
  exit 1
fi

FINETUNED_MODEL="$1"

echo "Setting up finetuned model: $FINETUNED_MODEL"

BASE_MODEL=$(grep -oP '/scratch/tmp\.[0-9]+\.rslocum/\K[a-zA-Z0-9._-]+' $SCRATCH/$FINETUNED_MODEL/adapter_config.json)

rsync -a --exclude 'checkpoint*/' "$SCRATCH/$FINETUNED_MODEL/" "$TMPDIR/finetuned/"

# Replace references to previous tmpdir drive to current scratch drive
grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/finetuned | while read -r file; do
  sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
done

if [[ -n "$RL_FINETUNED_MODEL" ]]; then
  rsync -a --exclude 'checkpoint*/' "$SCRATCH/rl_grpo_cutlist/$RL_FINETUNED_MODEL/" "$TMPDIR/rl_finetuned/"

  grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/rl_finetuned | while read -r file; do
    sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
  done
fi


cp $SCRATCH/$BASE_MODEL.tar $TMPDIR
tar -xf $TMPDIR/$BASE_MODEL.tar -C $TMPDIR/

cd ~/cutlist
source scripts/eulerScripts/setup_euler_python.sh