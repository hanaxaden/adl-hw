from homework.base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Format the question into a step-by-step reasoning template.
        The model should always produce a final answer inside <answer>...</answer>.
        """
    messages = [
    {"role": "system", "content": "You are a helpful assistant. Solve unit conversions step by step."},
    {"role": "user", "content": "Convert 5 meters to feet."},
    {"role": "assistant", "content": "1 meter = 3.28084 feet. 5 * 3.28084 = <answer>16.4042</answer>"}
    ]
    
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(prompt_str)


        # Apply chat template using the tokenizer
        ##prompt = self.tokenizer.apply_chat_template(
            ##messages,
            ##add_generation_prompt=True,
            ##tokenize=False
        ##)
        ##return prompt
##

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
