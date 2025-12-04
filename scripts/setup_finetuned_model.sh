if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <model_dir in scratch (e.g. model_output/48059951/checkpoint-22000)> "
  exit 1
fi

FINETUNED_MODEL="$1"

echo "Setting up finetuned model: $FINETUNED_MODEL"

BASE_MODEL=$(grep -oP '/scratch/tmp\.[0-9]+\.rslocum/\K[a-zA-Z0-9._-]+' $SCRATCH/$FINETUNED_MODEL/adapter_config.json)

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="Llama-3.2-1B-Instruct"
  echo "No base model detected -- using default: $BASE_MODEL"
fi

rsync -a --exclude 'checkpoint*/' "$SCRATCH/$FINETUNED_MODEL/" "$TMPDIR/finetuned/"

# Replace references to previous tmpdir drive to current scratch drive
grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/finetuned | while read -r file; do
  sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
done


cp $SCRATCH/$BASE_MODEL.tar $TMPDIR
tar -xf $TMPDIR/$BASE_MODEL.tar -C $TMPDIR/

cd ~/cutlist
source scripts/setup_euler_python.sh
