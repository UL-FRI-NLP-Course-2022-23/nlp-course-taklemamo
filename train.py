import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from data import ParaphraseDataset, max_len, padding_idx
from model import TransformerModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
n_epochs = 2
learning_rate = 1e-3
batch_size = 32

# model hyperparams
emb_size = 128
n_heads = 4
n_layers = 2
seq_len = max_len

# data
train_set = ParaphraseDataset("data/test_data.csv")
loader = DataLoader(train_set, batch_size, shuffle=True)
vocab_size = train_set.vocab_size


def train_step(model, optim, criterion):
    model.train()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optim.zero_grad() # reset the gradients
        preds = model(inputs, targets) # forward pass
        loss = criterion(preds, targets)
        loss.backward()
        optim.step() # backward pass


if __name__ == "__main__":
    model = model = TransformerModel(
        vocab_size,
        emb_size,
        n_heads,
        n_layers,
        seq_len,
        device
    ).to(device)
    optim = Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    test_sentence = "I am eating cheese."

    for epoch in range(n_epochs):
        print(f"{epoch+1}/{n_epochs}")
        train_step(model, optim, criterion)

