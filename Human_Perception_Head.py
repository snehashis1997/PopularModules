import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return self.norm(attn_output + query)  # Residual connection

class HumanPerceptionHead(nn.Module):
    def __init__(self, feature_dim=768, num_joints=55, hidden_dim=512, num_heads=8):
        super().__init__()
        self.pose_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3)  # Predicting joint rotations (axis-angle)
        )
        self.shape_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # SMPL-X shape parameters (betas)
        )
        self.cross_attn = CrossAttention(dim=feature_dim, num_heads=num_heads)
    
    def forward(self, person_features, global_features):
        """
        person_features: (batch, num_people, feature_dim)
        global_features: (batch, 1, feature_dim)  # Scene-level context
        """
        refined_features = self.cross_attn(person_features, global_features, global_features)
        pose_params = self.pose_mlp(refined_features)
        shape_params = self.shape_mlp(refined_features)
        return pose_params, shape_params

# Example usage
batch_size, num_people, feature_dim = 4, 5, 768
global_feature_dim = feature_dim

person_features = torch.randn(batch_size, num_people, feature_dim)
global_features = torch.randn(batch_size, 1, global_feature_dim)

hph = HumanPerceptionHead()
pose_out, shape_out = hph(person_features, global_features)

print("Pose output shape:", pose_out.shape)  # (batch, num_people, 55*3)
print("Shape output shape:", shape_out.shape)  # (batch, num_people, 10)
