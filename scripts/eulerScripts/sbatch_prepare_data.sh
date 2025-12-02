#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name=prepare_data
#SBATCH --output=_jobs/prepare_data%j.out
#SBATCH -c 16
#SBATCH --tmp=512G

VERBOSE=0

echo "Starting job -- $(date)"

source ~/setup_github.sh
source ~/cutlist/scripts/eulerScripts/setup_euler_python.sh 

cd $TMPDIR

##### Part One: Clone and unzip the data #####

if [ ! -d "StableText2Brick" ]; then
    echo "StableText2Brick directory does not exist. Cloning repository... $(date)"
    git clone https://huggingface.co/datasets/AvaLovelace/StableText2Brick
    echo "Clone complete $(date)"
else
    echo "StableText2Brick directory already exists. Skipping clone. $(date)"
fi


if [ ! -d "PartNet-archive" ]; then
    echo "PartNet directory does not exist. Cloning repository... $(date)"
    git clone git@hf.co:datasets/ShapeNet/PartNet-archive
    echo "Clone complete $(date)"
    cd PartNet-archive
else
    echo "PartNet directory already exists. Skipping clone. $(date)"
    cd PartNet-archive
fi



# Extract the multi-part archive
if [ -f "data_v0_chunk.zip" ] && [ ! -d "data_v0" ]; then
    echo "Extracting PartNet data_v0 multi-part archive... $(date)"
    zip -s 0 data_v0_chunk.zip --out data_v0_combined.zip
    unzip -q data_v0_combined.zip
    rm data_v0_combined.zip
    echo "PartNet data_v0 extraction complete. -- $(date)"
    echo "Size of extracted data_v0:"
    du -sh data_v0
else
    echo "PartNet data_v0 already extracted or archive not found. Skipping extraction. $(date)"
fi

# Extract additional archives if needed
if [ -f "ins_seg_h5.zip" ] && [ ! -d "ins_seg_h5" ]; then
    echo "Extracting ins_seg_h5... $(date)"
    unzip -q ins_seg_h5.zip
    echo "Extraction of ins_seg_h5 complete. -- $(date)"
fi

if [ -f "sem_seg_h5.zip" ] && [ ! -d "sem_seg_h5" ]; then
    echo "Extracting sem_seg_h5... $(date)"
    unzip -q sem_seg_h5.zip
    echo "Extraction of sem_seg_h5 complete. -- $(date)"
fi

##### Part Two: Clean the data by removing unwanted categories #####


# Define valid categories
valid_categories=("Bed" "Chair" "Table" "StorageFurniture")

# Function to check if a category is valid
is_valid_category() {
    local category="$1"
    for valid_cat in "${valid_categories[@]}"; do
        if [[ "$category" == "$valid_cat" ]]; then
            return 0
        fi
    done
    return 1
}

# Get the directory to scan (default to current directory)
SCAN_DIR=$TMPDIR/PartNet-archive/data_v0

# Counter for deleted folders
deleted_count=0

echo "Scanning directory: $SCAN_DIR"
echo "Valid categories: ${valid_categories[*]}"
echo "--- $(date) ---"

# Loop through each subdirectory
for folder in "$SCAN_DIR"/*; do
    # Skip if not a directory
    if [[ ! -d "$folder" ]]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    meta_file="$folder/meta.json"
    
    # Check if meta.json exists
    if [[ ! -f "$meta_file" ]]; then
        echo "No meta.json found in $folder_name - DELETING"
        rm -rf "$folder"
        ((deleted_count++))
        continue
    fi
    
    model_cat=$(grep -o '"model_cat"[[:space:]]*:[[:space:]]*"[^"]*"' "$meta_file" 2>/dev/null | sed 's/.*"model_cat"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    
    # Check if model_cat was found and is valid
    if [[ -z "$model_cat" ]]; then
        echo "No model_cat found in $folder_name/meta.json - DELETING"
        rm -rf "$folder"
        ((deleted_count++))
    elif is_valid_category "$model_cat"; then
        log "Valid category '$model_cat' in $folder_name - KEEPING"
    else
        log "Invalid category '$model_cat' in $folder_name - DELETING"
        rm -rf "$folder"
        ((deleted_count++))
    fi
done

echo "--- $(date) ---"
echo "Deleted $deleted_count folders"


echo "Size of cleaned data_v0:"
du -sh data_v0



# Create an uncompressed tar archive of the cleaned data for partnet
echo "Archiving cleaned data (no compression) -- $(date)"
tar -cf cleaned_data_v0.tar data_v0
echo "Archiving complete -- $(date)"

echo "Copying cleaned data back to $SCRATCH/data/ -- $(date)"
cp cleaned_data_v0.tar $SCRATCH/data/
    


##### Part Three: Prepare data for fine-tuning #####

echo "Running generation script -- $(date)"

uv run ./generate_data.py \
    --partnet_dir $TMPDIR/data_v0 \
    --brickgpt_dir $TMPDIR/StableText2Brick/data \
    --output $TMPDIR/

echo "Copying data back -- $(date)"
cp -r $TMPDIR/finetuning_data $SCRATCH/data/

echo "Job completed -- $(date)"



