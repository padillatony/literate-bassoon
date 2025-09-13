
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

MODEL_PATH = "distilgpt2"  # Cambia por tu modelo base o ruta a .safetensors
OUTPUT_PATH = "./savant_finetuned"

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset de ejemplo (puedes cambiarlo por el tuyo en español)
    dataset = load_dataset("yelp_review_full")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_dir="./logs",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].select(range(1000)) # subset de prueba
    )

    trainer.train()
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

if __name__ == "__main__":
    train()
