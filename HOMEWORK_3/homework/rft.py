from .base_llm import BaseLLM
from .sft import test_model

from .data import Dataset, benchmark
from pathlib import Path
import json
import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel




def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

##copied over from sft.py
def tokenize(tokenizer, question: str, answer: str):
    """
    1) Append <EOS> to the question-answer pair.
    2) Tokenize with padding/truncation.
    3) Build `labels` where question tokens are masked (-100)
       and only the answer tokens are supervised.
    """
    eos = tokenizer.eos_token
    full_text = f"{question} {answer}{eos}"

    tokenizer.padding_side = "right" #right padding 
    tokenizer.pad_token = eos
    enc = tokenizer(full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=128)

    # mask questions 
    q_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * q_len + enc["input_ids"][q_len:]
    

    labels = [
        -100 if m == 0 else l
        for l, m in zip(labels, enc["attention_mask"])
    ]

    enc["labels"] = labels
    return enc


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Given a raw (question, numeric_answer) pair,
    format the answer as <answer>…</answer>.
    """
    ans = f"{float(answer):.1f}"
    return {
        "question": prompt.strip(),
        "answer": f"<answer>{ans}</answer>"
    }

#copied from sft.py
class TokenizedDataset:
    """
    Takes any dataset yielding (q, a) and a format_fn,
    applies formatting + tokenization to each example.
    """
    def __init__(self, tokenizer, data, format_fn):
        self.tokenizer = tokenizer
        self.data = data
        self.format_fn = format_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        fe = self.format_fn(q, a)
        return tokenize(self.tokenizer, **fe)

#train model
def train_model(
    output_dir: str,
    learning_rate: float = 5e-4,
    num_train_epochs: int = 5,
    per_device_eval_batch_size: int = 8,
):
    #load model
    llm = BaseLLM()
    tokenizer = llm.tokenizer
    backbone = llm.model
    #set up lora congig 
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        bias="none",
        target_modules="all-linear",
        lora_dropout=0.1,
    )
    #converts the base model into lora model 
    model = get_peft_model(backbone, lora_config) 
    model.enable_input_require_grads()
    model.print_trainable_parameters()
     
     #training arguments 
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        learning_rate= learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_eval_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_checkpointing=True,
        report_to="tensorboard",  
    )
    #tokens 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(tokenizer, Dataset("train"), format_example),
        eval_dataset=TokenizedDataset(tokenizer, Dataset("valid"), format_example),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


def test_model(ckpt_path: str):
    """
    Load the RFT adapter and benchmark on the valid split.
    """
    testset = Dataset("valid")
    llm = BaseLLM()

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path)\
                     .to(llm.device)
    llm.model.eval()

    result = benchmark(llm, testset, 100)
    print(f"accuracy={result.accuracy}  answer_rate={result.answer_rate}")

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})