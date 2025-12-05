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

    # If there is no text, this is already terminal.
    # Otherwise, it's the start of a sequence, so not terminal.
    lines = assistant_text.splitlines(keepends=True) if assistant_text else []
    new_examples.append({"messages": new_msgs, "is_terminal": len(lines) == 0})

    if assistant_text == "":
        return new_examples

    # Build cumulative assistant content entries
    # We iterate up to len(lines) + 1 to include the case where the prompt is the full design
    for i in range(1, len(lines) + 1):
        cumulative = "".join(lines[:i]).strip() + "\n"
        new_msgs = non_assistant_msgs + [{"role": "assistant", "content": cumulative}]

        # If i == len(lines), we have included all lines in the prompt.
        # The expected behavior is to stop.
        is_terminal = i == len(lines)
        new_examples.append({"messages": new_msgs, "is_terminal": is_terminal})

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
            is_terminal = item.get("is_terminal", False)
            # keep original messages and add prompt key; trainer expects "prompt"
            converted.append({"prompt": msgs, "is_terminal": is_terminal})

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{split}.jsonl")
            # Write each expanded example as a JSON line (preserve unicode)
            with open(save_path, "w", encoding="utf-8") as f:
                for item in converted:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


def reward_function(completions, prompts, is_terminal, **kwargs):
    rewards = []

    for completion, prompt, term in zip(completions, prompts, is_terminal):
        # Extract the assistant's content from prompt and completion to find what was generated
        prompt_last_content = prompt[-1]["content"]
        completion_last_content = completion[-1]["content"]

        # The generated text is the difference
        if completion_last_content.startswith(prompt_last_content):
            generated_text = completion_last_content[len(prompt_last_content) :]
        else:
            generated_text = completion_last_content

        has_stopped = not generated_text.strip()

        # CASE 1: The design is finished (Ground Truth says STOP)
        if term:
            if has_stopped:
                rewards.append(1.0)  # Correctly stopped
            else:
                rewards.append(0.0)  # Failed to stop when it should have
            continue

        # CASE 2: The design is NOT finished (Ground Truth says CONTINUE)
        if has_stopped:
            rewards.append(0.0)  # Stopped prematurely
            continue

        # If we are here, the model generated a part and it was supposed to.
        # Now we evaluate the quality of that part.

        # Parse the new part
        new_part_text = generated_text.strip().splitlines()[0]
        new_part = ArbitraryCuboid.from_text(new_part_text)

        if new_part is None:
            # print("New part text was invalid:", repr(new_part_text))
            rewards.append(0.0)
            continue

        # Reconstruct original design from prompt
        original_design_text = prompt_last_content.strip()
        if not original_design_text:
            # Empty design (first part)
            original_design = WoodDesign(parts=[], design_type=ArbitraryCuboid)
        else:
            original_design = WoodDesign.from_txt(
                original_design_text, design_type=ArbitraryCuboid
            )

        if original_design is None:
            print("Original design could not be processed")
            rewards.append(0.0)
            continue

        reward, idcs, reward_string = reward_for_new_part(original_design, new_part)

        # Visualization logic (only for the first item in batch to avoid spam)
        if (
            kwargs.get("trainer_state")
            and kwargs["trainer_state"].global_step % 100 == 0
            and len(rewards) == 0
        ):
            # ...existing code...
            meshes = [design_part.get_mesh() for design_part in original_design.parts]
            if idcs is not None:
                colors = ["red" if i in idcs else "tan" for i in range(len(meshes))]
            else:
                colors = ["tan"] * len(meshes)
            image = visualize(
                meshes + [new_part.get_mesh()],
                colors=colors + ["blue"],
                opacities=[0.5] * (len(meshes) + 1),
                show_image=False,
                text=f"Reward {reward:.4f} \n {reward_string}",
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

    training_args = GRPOConfig(
        output_dir=MODEL_OUTPUT_DIR,
        max_completion_length=18,
        beta=0.8,  # KL coefficient: penalizes deviation from the reference model to prevent reward hacking
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(MODEL_OUTPUT_DIR)
