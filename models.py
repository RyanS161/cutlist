import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from generate_data import create_instruction

from design import WoodDesign, ArbitraryCuboid


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"  # Apple Silicon
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


class LLM:
    """
    A small wrapper class for a language model.
    """

    def __init__(
        self,
        model_name: str,
        device: str = get_device(),
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.kv_cache = None
        self.kv_cache_saved = None
        self.input_ids_cache = None
        self.input_ids_cache_saved = None

    def __call__(
        self,
        prompt: str | torch.Tensor | None = None,
        return_as_ids: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Generates text, given a prompt.
        """

        # If prompt is None, continue generation from previously generated tokens
        if prompt is None:
            prompt = self.input_ids_cache
        else:
            self.reset_cache()

        # If prompt is a string, encode it into token ids
        if isinstance(prompt, str):
            encoded_input = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded_input["input_ids"].to(self.device)
            attention_mask = encoded_input["attention_mask"].to(self.device)
        else:
            input_ids = prompt.to(self.device)
            attention_mask = torch.ones_like(input_ids)

        # Run generation
        output_dict = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            num_return_sequences=1,
            past_key_values=self.kv_cache,
            return_dict_in_generate=True,
            **kwargs,
        )
        self.input_ids_cache = output_dict["sequences"]

        # Return result as token ids or as a string
        input_length = input_ids.shape[1]
        result_ids = output_dict["sequences"][0][input_length:]
        result = result_ids if return_as_ids else self.tokenizer.decode(result_ids)

        return (result, output_dict) if return_dict else result

    def reset_cache(self) -> None:
        self.kv_cache = DynamicCache()

    def save_state(self) -> None:
        self.kv_cache_saved = copy.deepcopy(self.kv_cache)
        self.input_ids_cache_saved = self.input_ids_cache

    def rollback_to_saved_state(self) -> None:
        self.kv_cache = self.kv_cache_saved
        self.input_ids_cache = self.input_ids_cache_saved


class Cutlist:
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        self.max_parts = 40
        self.temperature = 0.6
        # self.temperature_increase = cfg.temperature_increase
        # self.max_temperature = cfg.max_temperature
        self.top_k = 20
        self.top_p = 1.0
        self.device = get_device() if device == "auto" else device

        self.instruction_fn = create_instruction

        self.llm = LLM(self.model_name_or_path, self.device)

    def __call__(self, caption: str):
        design = self.generate_design(caption)
        return design

    def generate_design(self, caption: str) -> WoodDesign:
        design = WoodDesign([], ArbitraryCuboid)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": create_instruction(caption)},
        ]
        prompt = self.llm.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        for i in range(self.max_parts):
            self.llm.save_state()
            part_txt = self.generate_part(prompt)
            print(f"Generated part {i}:", part_txt)
            if not part_txt:
                print("End of design generation.")
                break
            try:
                part = ArbitraryCuboid.from_text(part_txt)
            except Exception as e:
                print("Error parsing part:", e)
                self.llm.rollback_to_saved_state()
                continue

            # If we successfully parsed a part, we can add it to the design
            design.add_part(part)

            # Update prompt with the new part

            new_messages = messages + [
                {"role": "assistant", "content": design.to_txt()},
            ]
            prompt = self.llm.tokenizer.apply_chat_template(
                new_messages, continue_final_message=True, return_tensors="pt"
            )

        return design

    def generate_part(self, prompt: str):
        result_ids = self.llm(
            prompt,
            return_as_ids=True,
            max_new_tokens=18,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        print("Raw generated ids:", result_ids)
        if self.llm.tokenizer.eos_token_id in result_ids:
            return None

        return self.llm.tokenizer.decode(result_ids, skip_special_tokens=True)
