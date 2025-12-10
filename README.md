# Generative AI for Woodworking Designs

The code in this repository is intended to train models that create woodworking designs for robotic assembly. There are a few main stages: dataset creation, training models, performing inference, and evaluating the designs.

## Instructions to replicate results

All code here is written to be run on the Euler Cluster (due to the large size of the data and models involved)

### Initial step:

Clone this repository in the home directory: `cd ~ && git clone git@github.com:RyanS161/cutlist.git`

### Data generation

Before you can run the data generation script, make sure you have completed the following:

- Requested access to the [StableBrick2Text dataset](https://huggingface.co/datasets/AvaLovelace/StableText2Brick)
- Requested access to the [PartNet dataset](https://huggingface.co/datasets/ShapeNet/PartNet-archive)
- Generated a [Hugging Face authentication key](https://huggingface.co/docs/hub/en/security-tokens) and saved it at `~/hf_token.txt`

Once these steps are complete, run `sbatch ~/cutlist/scripts/sbatch_prepare_data.sh` to generate the finetuning data. It will be saved at `$SCRATCH/data/finetuning_data`


### Running SFT

Assuming you have the finetuning data saved at `$SCRATCH/data/finetuning_data`, you can now train the SFT model.

Prereqs for this step:

- Request access to the [Llama-3.2-1B-Instruct model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- Clone the model: `cd ~ && hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ~`
- Tar and move to scratch: `tar -cf Llama-3.2-1B-Instruct.tar ~/Llama-3.2-1B-Instruct && cp ~/Llama-3.2-1B-Instruct.tar $SCRATCH`

Now you can run the SFT script: `sbatch ~/cutlist/scripts/sbatch_finetuning.sh`

### Running GRPO

When running the SFT script, make note of the SBATCH job id. This serves as the model ID for the SFT model that we will be further fine-tuning on.

With this job id, run the following script to start GRPO tuning: `sbatch ~/cutlist/scripts/sbatch_rl_grpo.sh JOB_NAME model_output/SFT_SBATCH_JOB_ID`


### Running Inference:

Now you can run inference and generate results for the series of prompts defined in `experiments.csv`

To run inference, first set up the desired model: `~/cutlist/scripts/setup_finetuned_model.sh model_output/SFT_SBATCH_JOB_ID` for the SFT model or `~/cutlist/scripts/setup_finetuned_model.sh rl_grpo_cutlist/JOB_NAME` for the GRPO model.

Then simple run the inference script! `cd ~/cutlist && uv run infer.py --model_name $TMPDIR/finetuned --input_prompts ~/experiments.csv`

Designs will be saved to `./designs`


### Scoring and results

Create a directory called `~/experiment_outputs`. Each output of inference or the web tool can be put here under a different model name (i.e. `sft` or `grpo`), and each of these folders should contain the designs produced by the experiments sheet (`sft/000`, `sft/001`, etc).

To rate with gemini API: `python collect_results.py ./experiment_outputs --gemini YOUR_API_KEY`

To rate with web inference: `python collect_results.py ./experiment_outputs`

To generate graphs and analytics: `python collect_results.py ./experiment_outputs --analyze`


## Project Structure:


#### `binvox_rw.py`

Helper script for processing parts from shapenet. Copied from `https://github.com/dimatura/binvox-rw-py`

#### `collect_results.py`:

Script to evaluate designs, both using a web-based rating platform and by calling the gemini API. Also creates useful graphs and visuals.

#### `design.py`:

Script defining the different types of wooden blocks in the project, and helper methods to visualize and go back and forth from text.

#### `extern_datasets.py`:

Script with helper functions to process StableBrick and PartNet models.

#### `generate_data.py`:

Script that generates the dataset used in this project by fitting cuboids to parts in partnet. Also creates fine-tuning data using the included prompt format.

#### `infer.py`:

A script that allows inference to be run over a series of prompts from the trained models.

#### `misc_helper.py`:

Helper script for processing parts from shapenet.

#### `models.py`:

Definition of LLM and Cutlist models that are used at inference time

#### `scoring.py`:

Script containing definition of reward function used for training with GRPO.

#### `train_grpo.py`:

Script that creates the RL dataset, then uses the TRL library to train the model using GRPO


#### `scripts/finetune.sh`

Script that starts the SFT process using the TRL library.

#### `scripts/sbatch_finetune.sh`

Script starting an sbatch job with everything necessary for SFT.

#### `scripts/sbatch_prepare_data.sh`

Script starting an sbatch job to prepare and generate the dataset used in this project.

#### `scripts/sbatch_rl_grpo.sh`

Script starting an sbatch job for training with GRPO.

#### `scripts/sbatch_euler_python.sh`

Script reused in many other scripts to setup the euler cluster for running python.

#### `scripts/setup_finetuned_model.sh`

Script reused in other scripts to move fine-tuned models from $SCRATCH to the $TMPDIR

## License:

This project is under the MIT licence

## Acknowledgements:

AI was used to generate portions of the code in this project.

## Contact:

If you find an issue or bug, please feel free to open an issue on the Github Page.