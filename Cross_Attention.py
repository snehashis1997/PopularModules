import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class CrossAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        Q = self.W_Q(x1)  # Queries from input 1
        K = self.W_K(x2)  # Keys from input 2
        V = self.W_V(x2)  # Values from input 2

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x1.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# Two different inputs
x1 = torch.rand(1, 4, 8)  # Query (e.g., text embeddings)
x2 = torch.rand(1, 4, 8)  # Key/Value (e.g., image embeddings)

cross_attention = CrossAttention(embed_size=8)
output, attn_weights = cross_attention(x1, x2)

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(attn_weights[0].detach().numpy(), annot=True, cmap="Greens")
plt.title("Cross-Attention Weights")
plt.show()
