
import torch
from DETR_2D_Object_Detection import DETR
import torch.nn as nn

class DETR3D(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=6):
        super(DETR3D, self).__init__()
        self.detr = DETR(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

    def forward(self, images, camera_matrices):
        class_logits, bboxes_2d = self.detr(images)

        # Convert 2D bounding boxes to 3D using camera projection
        bboxes_3d = self.project_to_3d(bboxes_2d, camera_matrices)
        return class_logits, bboxes_3d

    def project_to_3d(self, bboxes_2d, camera_matrices):
        # Apply inverse projection using camera matrices (simplified)
        # Random rotation matrices (B, 3, 3) ==== camera_matrices
        # Random 3D points (B, 3, 1) ====  bboxes_2d

        bboxes_3d = torch.bmm(torch.inverse(camera_matrices), bboxes_2d.unsqueeze(-1)).squeeze(-1)
        return bboxes_3d
