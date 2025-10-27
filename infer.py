import time
import transformers
from models import LLM
from generate_data import create_instruction
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a language model."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The name of the model to use."
    )
    args = parser.parse_args()

    llm = LLM(model_name=args.model_name)

    caption = input("Enter a prompt, or <Return> to exit: ")

    while True:
        if not caption:
            break

        seed = input("Enter a generation seed (default=42): ")
        seed = int(seed) if seed else 42
        transformers.set_seed(seed)

        # Generate bricks
        print("Generating...")
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": create_instruction(caption)},
        ]
        prompt = llm.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        result_ids = llm(
            prompt,
            return_as_ids=True,
            max_new_tokens=1000,
            temperature=0.6,
            top_k=20,
            top_p=1.0,
        )

        output = llm.tokenizer.decode(result_ids, skip_special_tokens=True)
        end_time = time.time()

        print("Generated output:", output)
        print("Generation time:", end_time - start_time)

        caption = input("Enter another prompt, or <Return> to exit: ")


if __name__ == "__main__":
    main()
