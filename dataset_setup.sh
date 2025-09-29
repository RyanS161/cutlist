
if [ ! -d "ShapeNetCore" ]; then
    echo "ShapeNetCore directory does not exist. Cloning repository..."
    git clone git@hf.co:datasets/ShapeNet/ShapeNetCore
    cd ShapeNetCore
else
    echo "ShapeNetCore directory already exists. Skipping clone."
    cd ShapeNetCore
    git pull
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