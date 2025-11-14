from .base_llm import BaseLLM



class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {"role": "system",    "content": "You will perform unit conversions and be concise."},
            
            # Example D: temperature
            {"role": "user",      "content": "Convert 68°F to Celsius."},
            {"role": "assistant", "content": "C = (F - 32) * 5/9. (68 - 32) * 5/9 = <answer>20</answer>"},

            # Example E: volume
            {"role": "user",      "content": "Please convert 500 milliliters into liters."},
            {"role": "assistant", "content": "1 L = 1000 mL. 500 / 1000 = <answer>0.5</answer>"},

            # Example F: speed
            {"role": "user",      "content": "Could you convert 30 m/s to km/h?"},
            {"role": "assistant", "content": "1 m/s = 3.6 km/h. 30 * 3.6 = <answer>108</answer>"},

            # actual query placeholder
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
