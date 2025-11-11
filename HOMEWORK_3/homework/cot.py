from homework.base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Format the question into a step-by-step reasoning template.
        The model should always produce a final answer inside <answer>...</answer>.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Always reason step by step before giving the final answer. "
                    "The final answer must be inside <answer>...</answer> tags, nothing else."
                )
            },
            # Example to show format
            {
                "role": "user",
                "content": "Q: How many centimeters are there in 2 meters?"
            },
            {
                "role": "assistant",
                "content": (
                    "Step 1: 1 meter = 100 centimeters.\n"
                    "Step 2: Multiply 2 × 100 = 200.\n"
                    "<answer>200</answer>"
                )
            },
            # Actual question for the model to answer
            {
                "role": "user",
                "content": f"Q: {question}"
            }
        ]

        # Apply chat template using the tokenizer
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
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
