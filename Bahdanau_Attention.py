
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden_state):
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_state.unsqueeze(1)))
        attention_weights = torch.nn.functional.softmax(self.V(score), dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)

        return context_vector, attention_weights

# Simulated input
encoder_outputs = torch.rand(1, 4, 8)  # (batch, seq_len, hidden_size)
hidden_state = torch.rand(1, 8)  # (batch, hidden_size)

bahdanau_attention = BahdanauAttention(hidden_size=8)
context_vector, attn_weights = bahdanau_attention(encoder_outputs, hidden_state)

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(attn_weights.squeeze().detach().numpy(), annot=True, cmap="Purples")
plt.title("Bahdanau Attention Weights")
plt.show()
