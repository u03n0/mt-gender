from transformers import MarianMTModel, MarianTokenizer


class Marian():
    def __init__(self, src, trg, ds, device):
        self.source = src
        self.target = trg
        self.dataset = ds
        self.device = device
        self.model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def translate(self, example):
        batch = self.tokenizer(example['source'],
                            truncation=True,
                            padding=True,
                            return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**batch)
        return {'translation': self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}