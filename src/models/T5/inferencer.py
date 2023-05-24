import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from typing import Literal

class T5Inferencer():

    def __init__(self, model_path, model_size: Literal["large", "small"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(f"cjvt/t5-sl-{model_size}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def generate(self, text_list):
        return [self._generate_single(text) for text in text_list]

    def _generate_single(self, sentence):

        text = "parafraziraj: " + sentence
        max_len = 512

        encoding = self.tokenizer.encode_plus(text, max_length=max_len, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        # TODO tinker with this
        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=max_len,
            do_sample=True,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_beams=3,
            num_return_sequences=1,
        )[0]
        para = self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return para
    