from .base_llm import BaseLLM
from .data import Dataset, benchmark
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a question/answer pair.
    Mask the question so the loss is only computed on the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    question_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * question_len + full["input_ids"][question_len:]

    # Mask padding
    labels = [l if m != 0 else -100 for l, m in zip(labels, full["attention_mask"])]
    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Convert a prompt and answer into the expected format for the LLM.
    """
    rounded_answer = str(round(float(answer), 4))
    return {
        "question": f"Q: {prompt}",
        "answer": f"A: <answer>{rounded_answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.tokenizer = tokenizer
        self.data = data
        self.format_fn = format_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **example)


def train_model(output_dir: str, **kwargs):
    llm = BaseLLM()

    # LoRA adapter
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # safer than "all-linear"
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    # Dataset
    train_data = Dataset("train")
    tokenized_train = TokenizedDataset(llm.tokenizer, train_data, format_example)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        logging_dir=output_dir,
        report_to="tensorboard",
        save_total_limit=1,
        logging_steps=10,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_train,
    )

    trainer.train()
    llm.model.save_pretrained(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
