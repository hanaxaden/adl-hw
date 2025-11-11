from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# HuggingFace checkpoint
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to the LLM.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse <answer></answer> tag from output and return float.
        Robust to missing or malformed tags.
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    @staticmethod
    def clean_output(output: str) -> str:
        """
        Remove repeated sentences and extra whitespace from model output.
        """
        sentences = re.split(r'(?<=[.!?])\s+', output)
        seen = set()
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                cleaned.append(s)
                seen.add(s)
        return " ".join(cleaned)

    def generate(self, prompt: str) -> str:
        """
        Generate a single string output from the model.
        """
        raw_output = self.batched_generate([prompt])[0]
        return self.clean_output(raw_output)

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of generate with optional multiple sequences.
        """
        from tqdm import tqdm

        # Prevent GPU OOM
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx:idx + micro_batch_size], num_return_sequences, temperature)
            ]

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        nrs = num_return_sequences or 1
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            num_return_sequences=nrs,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Only decode newly generated tokens
        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # Return nested lists if num_return_sequences > 1
        if nrs > 1:
            return [[self.clean_output(decoded[i + j]) for j in range(nrs)] for i in range(0, len(decoded), nrs)]
        return [self.clean_output(d) for d in decoded]

    def answer(self, *questions) -> list[float]:
        """
        Answer multiple questions using LLM.
        """
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


#  test code
if __name__ == "__main__":
    from fire import Fire

    def test_model():
        testset = ["The cat went up", "The dog went down"]
        model = BaseLLM()
        for t in testset:
            print("Input:", t)
            print("Output:", model.generate(t))

        answers = model.batched_generate(testset)
        print("Batched outputs:", answers)

    Fire({"test": test_model})
