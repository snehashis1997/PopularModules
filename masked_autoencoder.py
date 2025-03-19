import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)

        # Generate a mask (lower triangular matrix)
        seq_len = attn_scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        # Apply softmax and compute output
        attn_weights = self.softmax(attn_scores)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# Simulated input (batch_size=1, num_tokens=5, embed_size=8)
x = torch.rand(1, 5, 8)
masked_self_attention = MaskedSelfAttention(embed_size=8)
output, attn_weights = masked_self_attention(x)

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(attn_weights[0].detach().numpy(), annot=True, cmap="Oranges", xticklabels=[f"T{i}" for i in range(5)], yticklabels=[f"T{i}" for i in range(5)])
plt.title("Masked Self-Attention Weights")
plt.show()
