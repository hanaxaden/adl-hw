from homework.base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        ##creating messages for the chat template to use to answer the question
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise.Be clear."},
            {"role": "user", "content": "Q: How many centimeters are there in 2 meters?"},
            {"role": "assistant", "content": ("A: 1 meter = 100 centimeters. So 2 meters = 2 × 100 = 200 centimeters. ""<answer>200</answer>")},
            {"role": "user", "content": f"Q: {question}"},
             
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt

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
