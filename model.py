import torch
from torch import nn

class TransformerModel(nn.Module):

    def __init__(self, vocab_size, n_layers=2, emb_dim=128, n_heads=2):
        super().__init__()

        self.encoder = TransformerEncoder(n_layers, emb_dim, n_heads)
        self.decoder = TransformerDecoder(n_layers, emb_dim, n_heads)

        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = ... #TODO

    def forward(self, x, memory):
        #TODO
        pass


class TransformerEncoder(nn.Module):
    
    def __init__(self, n_layers, emb_dim, n_heads):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(emb_dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    
    def __init__(self, n_layers, emb_dim, n_heads) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x


class EncoderBlock(nn.Module):
    
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.att_block = AttentionSubBlock(emb_dim, n_heads)
        self.ff_block = FFSubBlock(emb_dim)

    def forward(self, x):
        x = self.att_block(x, x, x)
        x = self.ff_block(x)
        return x
        

class DecoderBlock(nn.Module):
    
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.mask_att_block = AttentionSubBlock(emb_dim, n_heads)
        self.encoder_att_block = AttentionSubBlock(emb_dim, n_heads)
        self.ff_block = FFSubBlock(emb_dim)

    def forward(self, x, memory):
        # x is the previous decoder pred
        # memory is the encoder signal
        # TODO mask the output
        x = self.mask_att_block(x, x, x)
        x = self.encoder_att_block(x, memory, memory)
        x = self.ff_block(x)


class AttentionSubBlock(nn.Module):

    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, q, k, v):
        skip = q
        x, weights = self.attention(q, k, v) # sussy mogoce je tukej ksn bug - glej decoder block
        x += skip
        x = self.norm(x)
        return x
    

class FFSubBlock(nn.Module):

    def __init__(self, emb_dim, ff_dim=256):
        super().__init__()
        self.ff1 = nn.Linear(emb_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, emb_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        skip = x
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        x += skip
        x = self.norm(x)
        return x


if __name__ == "__main__":
    input = torch.ones((1, 128))
    memory = torch.zeros_like(input)
    model = EncoderBlock(128, 2)
    with torch.no_grad():
        print(model(input))
