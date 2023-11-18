from transformers import AutoTokenizer, MarianMTModel

src = "eng"
trg = "fr"

model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)