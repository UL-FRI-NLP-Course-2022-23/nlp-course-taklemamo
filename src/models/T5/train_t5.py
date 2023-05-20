import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset, random_split
import pandas as pd


class ParaDataset(Dataset):

	def __init__(self, fpath, tokenizer, prefix, max_len=512):
		super().__init__()
		self.raw_data = self._load(fpath)
		self.tokenizer = tokenizer
		self.prefix = prefix
		self.max_len = max_len

		self.inputs, self.targets = self._preprocess()

	def __len__(self):
		return len(self.raw_data)

	def __getitem__(self, index):
		out = {k:v[index] for k,v in self.inputs.items()}
		out["labels"] = self.targets.input_ids[index]
		return out

	def _load(self, fpath):
		return pd.read_csv(fpath, sep="\t", names=["paragraph", "paraphrase"])

	def _preprocess(self):
		return self._tokenize(self.raw_data.paragraph), self._tokenize(self.raw_data.paraphrase, prefix=False)

	def _tokenize(self, text_list, prefix=True):
		return self.tokenizer(
			[self.prefix + text if prefix else text for text in text_list],
			truncation=True, padding="max_length", 
			max_length=self.max_len, return_tensors="pt"
			)
  


def main():
	output_dir = "/d/hpc/projects/FRI/DL/gs1121/NLP/t5"
	dataset_path = "../../../data/backtranslate/backtranslate_filtered.csv"

	# checkpoint = "cjvt/t5-sl-small"
	checkpoint = "cjvt/t5-sl-large"

	SEED = 1337
	max_len = 512 # num of input/output tokens

	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	tokenizer(["test"], return_tensors="pt")

	data = pd.read_csv(dataset_path, sep="\t", names=["inputs", "targets"])
	paraset = ParaDataset(dataset_path, tokenizer, "parafraziraj: ", max_len)

	gen = torch.Generator().manual_seed(SEED)
	train_set, val_set = random_split(paraset, [0.9, 0.1], generator=gen)


	training_args = TrainingArguments(
		output_dir= output_dir,
		overwrite_output_dir=True,
		save_strategy="epoch",
		save_total_limit=1,
		evaluation_strategy = "epoch",
		num_train_epochs=15,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		warmup_steps=500,
		#weight_decay=0.01,
		logging_steps=10,
		disable_tqdm=True,
		load_best_model_at_end=True,
		seed=SEED
	)

	model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
	data_collator = DataCollatorForSeq2Seq(tokenizer, model)

	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_set,
		eval_dataset=val_set
	)

	trainer.train()
	


if __name__ == "__main__":
	main()