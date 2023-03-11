import torch
from torch.utils.data import Dataset
from torch.hub import load_state_dict_from_url
import torchtext.transforms as T
from torchtext.vocab import Vocab

import pandas as pd


padding_idx = 1
bos_idx = 0
eos_idx = 2
max_len = 128
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

vocab = Vocab(load_state_dict_from_url(xlmr_vocab_path))

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(vocab),
    T.Truncate(max_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
    T.ToTensor(padding_idx),
    T.PadTransform(max_len, padding_idx),
)


class ParaphraseDataset(Dataset):
    
    def __init__(self, fpath, transform=text_transform):
        super().__init__()
        self.path = fpath
        self.transform = transform
        self.inputs, self.targets = self.__load_sentences()
        self.vocab_size = max(self.inputs.max(), self.targets.max())

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
    def __len__(self):
        return len(self.inputs)
    
    def __load_sentences(self):
        df = pd.read_csv(self.path, sep="\t",  names=["sentence", "paraphrase"])
        inputs = self.transform(df["sentence"].tolist())
        targets = self.transform(df["paraphrase"].tolist())

        return inputs, targets


def preprocess(fpath):
    df = pd.read_csv(fpath, sep="\t", usecols=[0, 2],  names=["id", "sentence"])
    
    sents = []
    next_sents = []
    for row, next_row in zip(df.iterrows(), df[1:].iterrows()):
        if row[1]["id"] == next_row[1]["id"]:
            sents.append(row[1]["sentence"])
            next_sents.append(next_row[1]["sentence"])

    df = pd.DataFrame({"input": sents, "target": next_sents})
    df.to_csv("data/out.csv", sep="\t", index=False, header=False)


if __name__ == "__main__":
    #preprocess("data/en.txt")
    dataset = ParaphraseDataset("data/test_data.csv", text_transform)
    print(dataset.__getitem__(0))
    print(dataset.vocab_size)
