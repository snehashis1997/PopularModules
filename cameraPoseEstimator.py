
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraPoseEstimator(nn.Module):
    def __init__(self, feature_dim=256):
        super(CameraPoseEstimator, self).__init__()
        
        # MLP to predict rotation (as a quaternion) and translation
        self.mlp_rotation = nn.Linear(feature_dim, 4)  # Quaternion (x, y, z, w)
        self.mlp_translation = nn.Linear(feature_dim, 3)  # Translation vector (x, y, z)

    def forward(self, feature_map):
        """
        feature_map: (batch_size, feature_dim)
        """
        rotation = F.normalize(self.mlp_rotation(feature_map), p=2, dim=-1)  # Normalize quaternion
        translation = self.mlp_translation(feature_map)

        return rotation, translation
