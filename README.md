All code here is written to be run on the Euler Cluster (due to the large size of the data involved)

## Initial step:

Clone this repository in the home directory: `cd ~ && git clone git@github.com:RyanS161/cutlist.git`

## Data generation

Before you can run the data generation script, make sure you have completed the following:

- Requested access to the [StableBrick2Text dataset](https://huggingface.co/datasets/AvaLovelace/StableText2Brick)
- Requested access to the [PartNet dataset](https://huggingface.co/datasets/ShapeNet/PartNet-archive)
- Generated a [Hugging Face authentication key](https://huggingface.co/docs/hub/en/security-tokens) and saved it at `~/hf_token.txt`


Once these steps are complete, run `sbatch ~/cutlist/scripts/sbatch_prepare_data.sh` to generate the finetuning data. It will be saved at `$SCRATCH/data/finetuning_data`
