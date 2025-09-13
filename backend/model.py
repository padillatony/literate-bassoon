
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

class SavantModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        print(f"⏳ Cargando modelo desde {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(device)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        print("✅ Modelo listo.")

    def infer(self, prompt, max_length=250):
        if self.pipeline is None:
            return "❌ Error: modelo no cargado."
        output = self.pipeline(
            prompt,
            max_length=max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return output[0]["generated_text"].replace(prompt, "").strip()
