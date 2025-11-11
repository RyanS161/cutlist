from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json

from scoring import score_new_part
from design import WoodDesign, ArbitraryCuboid
import argparse

os.environ["WANDB_ENTITY"] = "ryanslocum-eth-zurich"
os.environ["WANDB_PROJECT"] = "cutlist_rlft"


parser = argparse.ArgumentParser(
    description="Train GRPO for cutlist. Provide paths for data, model, adapter, and output."
)
parser.add_argument(
    "--finetuning-data-dir",
    default="~/finetuning_data",
    help="Path to the original fine-tuning dataset (HF dataset or dataset dir).",
)
parser.add_argument(
    "--rl-data-dir",
    default="~/rl_tuning_data",
    help="Directory to write/read expanded RL dataset (JSONL per split).",
)
parser.add_argument(
    "--base-model-path",
    default="~/Llama-3.2-1B-Instruct",
    help="Path or model id of the base causal LM.",
)
parser.add_argument(
    "--adapter-path",
    default="~/finetuned",
    help="Path to the PEFT/LoRA adapter weights.",
)
parser.add_argument(
    "--model-output-dir",
    default="~/rl_grpo_cutlist",
    help="Directory where the trained GRPO model and artifacts will be saved.",
)

args = parser.parse_args()

# Expand tildes and convert to absolute paths
FINETUNING_DATA_DIR = os.path.abspath(os.path.expanduser(args.finetuning_data_dir))
RL_DATA_DIR = os.path.abspath(os.path.expanduser(args.rl_data_dir))
BASE_MODEL_PATH = os.path.abspath(os.path.expanduser(args.base_model_path))
ADAPTER_PATH = os.path.abspath(os.path.expanduser(args.adapter_path))
MODEL_OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.model_output_dir))


# Replace inline expansion logic with reusable functions
def expand_example(example):
    """
    Take one example (with example['messages']) and return a list of new examples:
    - first: assistant content = ""
    - then cumulative assistant content up to each newline (keeps newline chars)
    Other message roles are copied as-is; multiple assistant messages are concatenated.
    """
    msgs = example.get("messages", []) or []
    # Concatenate all assistant message contents into one string
    assistant_text = "".join(
        [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    )

    # Prepare the base non-assistant messages and a template for inserting assistant content
    non_assistant_msgs = [dict(m) for m in msgs if m.get("role") != "assistant"]

    new_examples = []

    # First entry: assistant content is empty
    new_msgs = non_assistant_msgs + [{"role": "assistant", "content": ""}]
    new_examples.append({"messages": new_msgs})

    if assistant_text == "":
        return new_examples

    # Split into lines keeping newline characters
    lines = assistant_text.splitlines(keepends=True)

    # Build cumulative assistant content entries
    for i in range(1, len(lines)):
        cumulative = "".join(lines[:i]).strip()
        new_msgs = non_assistant_msgs + [{"role": "assistant", "content": cumulative}]
        new_examples.append({"messages": new_msgs})

    return new_examples


def expand_dataset(hf_dataset):
    """Expand every example in the provided HF dataset (split) and return a list of expanded examples."""
    expanded = []
    for ex in hf_dataset:
        expanded.extend(expand_example(ex))
    return expanded


def expand_and_save(dataset_path, splits=("train", "test"), output_dir=None):
    """
    Load each split from dataset_path, expand examples, optionally save expanded split to output_dir
    as JSONL (one JSON object per line), and return a dict of expanded HF Datasets keyed by split name.
    """
    for split in splits:
        try:
            ds = load_dataset(dataset_path, split=split)
        except Exception as e:
            # Skip missing splits but continue
            print(f"Warning: unable to load split '{split}': {e}")
            continue

        expanded_list = expand_dataset(ds)

        converted = []
        for item in expanded_list:
            msgs = item.get("messages", [])
            # keep original messages and add prompt key; trainer expects "prompt"
            converted.append({"prompt": msgs})

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{split}.jsonl")
            # Write each expanded example as a JSON line (preserve unicode)
            with open(save_path, "w", encoding="utf-8") as f:
                for item in converted:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Configuration: original dataset source and where to save expanded datasets

# Expand both train and test and save expanded versions; get expanded train for training
expand_and_save(FINETUNING_DATA_DIR, splits=("train", "test"), output_dir=RL_DATA_DIR)

dataset = load_dataset(RL_DATA_DIR, split="train")


# Dummy reward function for demonstration purposes
def reward_function(completions, prompts, **kwargs):
    scores = []
    for completion in completions:
        design_text = completion[0]["content"]
        try:
            design = WoodDesign.from_txt(design_text, design_type=ArbitraryCuboid)
        except Exception as e:
            print(f"Error processing design: {e}")
            print("Design text was:", repr(design_text))
            score = -1.0
            design = None

        if design:
            recent_part = design.parts.pop()  # get the most recently added part
            raw_score, idx = score_new_part(design, recent_part)
            score = -1.0 * abs(raw_score)

        print("Score for new part:", score)
        scores.append(score)

    return scores


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map=None)
# attach LoRA adapter weights
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

training_args = GRPOConfig(output_dir=MODEL_OUTPUT_DIR, max_completion_length=18)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
