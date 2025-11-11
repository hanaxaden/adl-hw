from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {"role": "system",    "content": "You will perform unit conversions and be concise."},
            
            # Example 1: speed
            {"role": "user", "content": "Can you convert 5 meters to feet?"},
            {"role": "assistant", "content": "1 meter = 3.28084 ft. 5 * 3.28084 = <answer>16.4042</answer>"},

            
            # Example 2: volume
            {"role": "user",      "content": "Could you convert 1 gallons to liters?"},
            {"role": "assistant", "content": "1 gallon = 3.78541 L. 1 * 3.78541 = <answer>3.78541</answer>"},
            
            # Example 3: time
            {"role": "user", "content": "How many grams are there in 6 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 g. 6 * 1000 = <answer>6000</answer>"},

            
            # actual query
            {"role": "user",      "content": question},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize = False,
        )


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
