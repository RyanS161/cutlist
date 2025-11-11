if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model_dir in model_output (e.g. 48059951/checkpoint-22000)>"
  exit 1
fi

FINETUNED_MODEL="$1"

BASE_MODEL=$(grep -oP '/scratch/tmp\.[0-9]+\.rslocum/\K[a-zA-Z0-9._-]+' $SCRATCH/model_output/$FINETUNED_MODEL/adapter_config.json)

rsync -av --exclude 'checkpoint*/' "$SCRATCH/model_output/$FINETUNED_MODEL/" "$TMPDIR/finetuned/"

# cp -r $SCRATCH/model_output/$FINETUNED_MODEL $TMPDIR/finetuned/

# Replace references to previous tmpdir drive to current scratch drive
grep -rIl -E '/scratch/tmp\.[0-9]+\.rslocum' $TMPDIR/finetuned | while read -r file; do
  sed -i -E "s|/scratch/tmp\.[0-9]+\.rslocum|${TMPDIR}|g" "$file"
done

cp $SCRATCH/$BASE_MODEL.tar $TMPDIR
tar -xf $TMPDIR/$BASE_MODEL.tar -C $TMPDIR/

cd ~/cutlist
source scripts/eulerScripts/setup_euler_python.sh