#!/bin/bash

# DATASET="ShapeNetCore"
DATASET="PartNet"

if [ "$DATASET" == "ShapeNetCore" ]; then
    #### FOR SHAPENET CORE ####
    if [ ! -d "ShapeNetCore" ]; then
        echo "ShapeNetCore directory does not exist. Cloning repository..."
        git clone git@hf.co:datasets/ShapeNet/ShapeNetCore
        cd ShapeNetCore
        rm -rf .git
    else
        echo "ShapeNetCore directory already exists. Skipping clone."
        cd ShapeNetCore
    fi

    ids=(
        02843684 04379243 03001627 02818832 02828884 02933112 04460130 02871439 02801938 03710193 03797390
    )

    # cabinet and mailbox are not in brickgpt

    names=(
        birdhouse table chair bed bench cabinet tower bookshelf basket mailbox mug
    )

    for i in "${!ids[@]}"; do
        mv "${ids[$i]}.zip" "${names[$i]}.zip"
    done

    rm 0**.zip

    for name in "${names[@]}"; do
        unzip "$name.zip" && rm "$name.zip"
    done
elif [ "$DATASET" == "PartNet" ]; then
    #### FOR PARTNET ####
    if [ ! -d "PartNet-archive" ]; then
        echo "PartNet directory does not exist. Cloning repository..."
        git clone git@hf.co:datasets/ShapeNet/PartNet-archive
        cd PartNet-archive
        rm -rf .git
    else
        echo "PartNet directory already exists. Skipping clone."
        cd PartNet-archive
    fi

    # Extract the multi-part archive
    if [ -f "data_v0_chunk.zip" ] && [ ! -d "data_v0" ]; then
        echo "Extracting PartNet data_v0 multi-part archive..."
        zip -s 0 data_v0_chunk.zip --out data_v0_combined.zip
        unzip data_v0_combined.zip
        rm data_v0_combined.zip
        echo "PartNet data_v0 extraction complete."
    else
        echo "PartNet data_v0 already extracted or archive not found."
    fi
    
    # Extract additional archives if needed
    if [ -f "ins_seg_h5.zip" ] && [ ! -d "ins_seg_h5" ]; then
        echo "Extracting ins_seg_h5..."
        unzip ins_seg_h5.zip
    fi
    
    if [ -f "sem_seg_h5.zip" ] && [ ! -d "sem_seg_h5" ]; then
        echo "Extracting sem_seg_h5..."
        unzip sem_seg_h5.zip
    fi

fi