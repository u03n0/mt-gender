from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class Mbart():
  def __init__(self, src, trg, ds, device):
    self.source = src
    self.target = trg
    self.dataset = ds
    self.device = device
    self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
    self.tokenizer = MBartForConditionalGeneration.from_pretrained(self.model_name)
    self.model = MBart50TokenizerFast.from_pretrained(self.model_name, src_lang=f"{src}_XX", tgt_lang=f"{trg}_XX")

  def translate(self, example):
    """ Gets the translation of the example 'source' based
    on the target language desired.
    """
    self.tokenizer.src_lang = f"{self.src}_XX"
    encoded = self.tokenizer(example['source'],
                          padding=True,
                          truncation=True,
                          max_length=128,
                          return_tensors="pt").to(self.device)
    generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[f"{self.trg}_XX"])
    return {'translation': self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)}