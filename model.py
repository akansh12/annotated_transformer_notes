import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    
    def __init__(self, d_input: int, vocab_size: int):
        super().__init__()
        self.d_model = d_input
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_input)

    def forward(self, x):
        return self.embedding(x)
        