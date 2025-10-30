import time
import transformers
from models import Cutlist
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a language model."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The name of the model to use."
    )
    args = parser.parse_args()

    model = Cutlist(model_name_or_path=args.model_name)

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
    main()
