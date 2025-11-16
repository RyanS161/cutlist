from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import AutoPeftModelForCausalLM
import os
import json
import numpy as np

from models import get_device
from scoring import reward_for_new_part
from design import WoodDesign, ArbitraryCuboid, visualize
import argparse
import wandb


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


# Dummy reward function for demonstration purposes
def reward_function(completions, prompts, **kwargs):
    rewards = []

    original_design_text = "\n".join(completions[0][0]["content"].splitlines()[:-1])
    original_design = WoodDesign.from_txt(
        original_design_text, design_type=ArbitraryCuboid
    )
    if original_design is None:
        print("Design could not be processed, text was:", repr(original_design_text))
        return [0.0] * len(completions)

    for completion in completions:
        # design_text = completion[0]["content"]
        new_part_text = completion[0]["content"].splitlines()[-1]
        new_part = ArbitraryCuboid.from_text(new_part_text)
        if new_part is None:
            print("New part text was:", repr(new_part_text))
            rewards.append(0.0)
            continue

        reward, idx = reward_for_new_part(original_design, new_part)
        if kwargs["trainer_state"].global_step % 1e3 == 0:
            meshes = [design_part.get_mesh() for design_part in original_design.parts]
            if idx is not None:
                colors = ["red" if i == idx else "tan" for i in range(len(meshes))]
            else:
                colors = ["tan"] * len(meshes)
            image = visualize(
                meshes + [new_part.get_mesh()],
                colors=colors + ["blue"],
                opacities=[0.5] * (len(meshes) + 1),
                show_image=False,
                text=f"Reward {reward:.4f}",
            )
            try:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "reward_image": wandb.Image(
                                image, caption=f"Reward {reward:.4f}"
                            )
                        },
                        # step=int(kwargs["trainer_state"].global_step),
                    )
            except Exception as e:
                print("wandb image log failed:", e)

            # print("Reward for new part:", reward)
        rewards.append(reward)

    arr = np.array(rewards, dtype=float)
    print(
        f"REWARD BATCH mean={arr.mean():.4f} std={arr.std():.4f} sample={arr[:6].tolist()}"
    )

    return rewards


if __name__ == "__main__":
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

    parser.add_argument(
        "--run-name",
        default="cutlist_grpo_run",
        help="Name for the training run (for logging purposes).",
    )
    args = parser.parse_args()

    args.finetuning_data_dir = (
        "/Users/ryanslocum/Documents/current_courses/semesterProject/finetuning_data"
    )
    args.rl_data_dir = (
        "/Users/ryanslocum/Documents/current_courses/semesterProject/rl_tuning_data"
    )
    args.base_model_path = "/Users/ryanslocum/Documents/current_courses/semesterProject/Llama-3.2-1B-Instruct"
    args.adapter_path = (
        "/Users/ryanslocum/Documents/current_courses/semesterProject/finetuned"
    )
    args.model_output_dir = (
        "/Users/ryanslocum/Documents/current_courses/semesterProject/rl_grpo_cutlist"
    )
    args.run_name = "test-test"

    # Expand tildes and convert to absolute paths
    FINETUNING_DATA_DIR = os.path.abspath(os.path.expanduser(args.finetuning_data_dir))
    RL_DATA_DIR = os.path.abspath(os.path.expanduser(args.rl_data_dir))
    BASE_MODEL_PATH = os.path.abspath(os.path.expanduser(args.base_model_path))
    ADAPTER_PATH = os.path.abspath(os.path.expanduser(args.adapter_path))
    MODEL_OUTPUT_DIR = os.path.join(
        os.path.abspath(os.path.expanduser(args.model_output_dir)), args.run_name
    )
    # Configuration: original dataset source and where to save expanded datasets

    # Expand both train and test and save expanded versions; get expanded train for training
    expand_and_save(
        FINETUNING_DATA_DIR, splits=("train", "test"), output_dir=RL_DATA_DIR
    )

    dataset = load_dataset(RL_DATA_DIR, split="train").shuffle(seed=42)

    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_PATH, is_trainable=True
    )  # attach LoRA adapter weights

    device = get_device()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"MODEL DEVICE={device} TOTAL_PARAMS={total_params:,} TRAINABLE_PARAMS={trainable_params:,}"
    )
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name, tuple(p.shape))

    # os.environ["WANDB_ENTITY"] = "ryanslocum-eth-zurich"
    # os.environ["WANDB_PROJECT"] = "cutlist_rlft"

    wandb.init(
        entity="ryanslocum-eth-zurich", project="cutlist_rlft", name=args.run_name
    )

    training_args = GRPOConfig(output_dir=MODEL_OUTPUT_DIR, max_completion_length=18)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(MODEL_OUTPUT_DIR)
