import time
import transformers
from models import Cutlist
import argparse
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def automatic_inference(model, input_csv):
    prompts_df = pd.read_csv(input_csv)
    for idx, row in prompts_df.iterrows():
        caption = row["prompt"]
        seed = int(row["seed"]) if "seed" in row and not pd.isna(row["seed"]) else 42
        output_dir = (
            row["output_dir"]
            if "output_dir" in row and not pd.isna(row["output_dir"])
            else f"./designs/design_{idx}/"
        )
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        transformers.set_seed(seed)

        # Generate bricks
        print(f"Generating design {idx}...")
        start_time = time.time()
        design = model(caption)
        end_time = time.time()

        # save results
        with open(os.path.join(output_dir, "design.txt"), "w") as f:
            f.write(design.to_txt())

        design_image_path = os.path.join(output_dir, "design.png")
        design_gif_path = os.path.join(output_dir, "design.gif")

        design.visualize_img(filename=design_image_path, text=caption)
        design.visualize_gif(filename=design_gif_path)

        print("Saved results to", output_dir)
        print("Generation time:", end_time - start_time)


def main(model):
    caption = input("Enter a prompt, or <Return> to exit: ")

    while True:
        if not caption:
            break

        seed = input("Enter a generation seed (default=42): ")
        seed = int(seed) if seed else 42
        output_dir = input("Enter a directory to save the output (default=./designs): ")
        output_dir = output_dir if output_dir else "./designs/"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        transformers.set_seed(seed)

        # Generate bricks
        print("Generating...")
        start_time = time.time()
        design = model(caption)
        end_time = time.time()

        # save results
        with open(os.path.join(output_dir, "design.txt"), "w") as f:
            f.write(design.to_txt())

        design_image_path = os.path.join(output_dir, "design.png")
        design_gif_path = os.path.join(output_dir, "design.gif")

        design.visualize_img(filename=design_image_path, text=caption)
        design.visualize_gif(filename=design_gif_path)

        print("Saved results to", output_dir)
        print("Generation time:", end_time - start_time)

        caption = input("Enter another prompt, or <Return> to exit: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a language model."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The name of the model to use."
    )
    parser.add_argument(
        "--rl_model_adapter", type=str, required=True, help="The RL model adapter path."
    )
    parser.add_argument(
        "--input_prompts", type=str, help="CSV file with input prompts.", default=None
    )
    args = parser.parse_args()

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.rl_model_adapter:
        finetuned_model = PeftModel.from_pretrained(
            base_model, args.rl_model_adapter
        )  # attach LoRA adapter weights
        llm_model = finetuned_model
    else:
        llm_model = base_model

    model = Cutlist(model_name_or_path=None)

    model.llm.tokenizer = base_tokenizer
    model.llm.model = llm_model
    model.llm.model.to(model.llm.device)

    if args.input_prompts:
        automatic_inference(model, args.input_prompts)
    else:
        main(model)
