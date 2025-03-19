
import torch
from DETR3D import DETR3D
import torch.nn as nn

class PETR(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=6):
        super(PETR, self).__init__()
        self.detr3d = DETR3D(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

        # 3D Positional Encoding
        self.positional_encoding_3d = nn.Linear(3, hidden_dim)

    def forward(self, images, camera_matrices, point_clouds):
        pos_3d = self.positional_encoding_3d(point_clouds)
        class_logits, bboxes_3d = self.detr3d(images, camera_matrices)

        return class_logits, bboxes_3d + pos_3d
