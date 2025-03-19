
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)
        self.W_O = nn.Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, num_tokens, embed_size = x.shape
        Q = self.W_Q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_size)

        return self.W_O(output), attention_weights.mean(dim=1)

# Multi-Head Attention Example
x = torch.rand(1, 4, 8)
multi_head_attention = MultiHeadAttention(embed_size=8, num_heads=2)
output, attn_weights = multi_head_attention(x)

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(attn_weights[0].detach().numpy(), annot=True, cmap="Oranges")
plt.title("Multi-Head Attention Weights")
plt.show()
