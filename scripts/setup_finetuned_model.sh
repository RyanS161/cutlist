if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <model_dir in model_output (e.g. 48059951/checkpoint-22000)> <rl_model_dir in model_output>"
  exit 1
fi

FINETUNED_MODEL="$1"
RL_FINETUNED_MODEL="${2:-}"

BASE_MODEL=$(grep -oP '/scratch/tmp\.[0-9]+\.rslocum/\K[a-zA-Z0-9._-]+' $SCRATCH/model_output/$FINETUNED_MODEL/adapter_config.json)

rsync -a --exclude 'checkpoint*/' "$SCRATCH/model_output/$FINETUNED_MODEL/" "$TMPDIR/finetuned/"

# Replace references to previous tmpdir drive to current scratch drive
grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/finetuned | while read -r file; do
  sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
done

if [[ -z "$RL_FINETUNED_MODEL" ]]; then
  rsync -a --exclude 'checkpoint*/' "$SCRATCH/rl_grpo_cutlist/$FINETUNED_MODEL/" "$TMPDIR/rl_finetuned/"

  grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/rl_finetuned | while read -r file; do
    sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
  done
fi


cp $SCRATCH/$BASE_MODEL.tar $TMPDIR
tar -xf $TMPDIR/$BASE_MODEL.tar -C $TMPDIR/

cd ~/cutlist
source scripts/eulerScripts/setup_euler_python.sh