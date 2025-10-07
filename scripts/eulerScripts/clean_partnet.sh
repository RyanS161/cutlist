#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name=unzip_data
#SBATCH --output=_jobs/unzip_data%j.out
#SBATCH -c 4
#SBATCH --tmp=512G
# Enable verbose mode if --verbose or -v is passed
VERBOSE=0
for arg in "$@"; do
    if [[ "$arg" == "--verbose" || "$arg" == "-v" ]]; then
        VERBOSE=1
        break
    fi
done

log() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "$@"
    fi
}

echo "Starting job -- $(date)"

cp -r $SCRATCH/data/PartNet-archive $TMPDIR/
cd $TMPDIR/PartNet-archive

echo "Extracting PartNet data_v0 multi-part archive -- $(date)"
zip -s 0 data_v0_chunk.zip --out data_v0_combined.zip
rm data_v0_chunk.z*
unzip -q data_v0_combined.zip
rm data_v0_combined.zip
echo "PartNet data_v0 extraction complete. -- $(date)"

# Print out the size of the extracted data
du -sh data_v0

# echo "Extracting ins_seg_h5..."
# unzip ins_seg_h5.zip

# echo "Extracting sem_seg_h5..."
# unzip sem_seg_h5.zip



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

du -sh data_v0

# Compress the cleaned data back into a zip file
echo "Compressing cleaned data -- $(date)"
zip -qr cleaned_data_v0.zip data_v0
echo "Compression complete -- $(date)"

echo "Copying cleaned data back to $SCRATCH/data/PartNet-archive/ -- $(date)"
cp cleaned_data_v0.zip $SCRATCH/data/PartNet-archive/


echo "Job completed -- $(date)"




