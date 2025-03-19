import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
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

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# Simulated input (batch_size=1, num_tokens=4, embed_size=8)
x = torch.rand(1, 4, 8)
self_attention = SelfAttention(embed_size=8)
output, attn_weights = self_attention(x)

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(attn_weights[0].detach().numpy(), annot=True, cmap="Blues")
plt.title("Self-Attention Weights")
plt.show()