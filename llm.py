import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx

def posneg_transform(x):
    return torch.maximum(torch.zeros_like(x), x) - torch.abs(torch.minimum(torch.zeros_like(x), x))

class ToyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        emb = posneg_transform(emb)
        hidden = self.linear(posneg_transform(emb))
        hidden = posneg_transform(hidden)
        activated = F.gelu(hidden)
        activated = posneg_transform(activated)
        q = self.Wq(posneg_transform(activated))
        k = self.Wk(posneg_transform(activated))
        v = self.Wv(posneg_transform(activated))
        attn_out = self.scaled_dot_product_attention(
            posneg_transform(q), posneg_transform(k), posneg_transform(v)
        )
        return attn_out

    def scaled_dot_product_attention(self, q, k, v):
        # Apply transform to q, k before matmul, and to the result before softmax
        q = posneg_transform(q)
        k = posneg_transform(k)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = posneg_transform(scores)
        d_k = q.size(-1)
        scores = scores / d_k**0.5
        attn = F.softmax(scores, dim=-1)
        attn = posneg_transform(attn)
        v = posneg_transform(v)
        out = torch.matmul(attn, v)
        out = posneg_transform(out)
        return out

# Example usage
vocab_size = 1000
embed_dim = 64
hidden_size = 128
batch_size = 2
seq_len = 10

model = ToyModel(vocab_size, embed_dim, hidden_size)
model.eval()
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Compile the model to MLIR using torch-mlir FX frontend
mlir_module = fx.export_and_import(model, token_ids)
print("MLIR Module:")
print(mlir_module)