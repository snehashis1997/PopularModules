import torch
import torch.nn.functional as F

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (B, 4) to a rotation matrix (B, 3, 3).
    q should be normalized before usage.
    """
    q = F.normalize(q, p=2, dim=-1)  # Ensure unit quaternion
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    B = q.shape[0]  # Batch size
    R = torch.zeros(B, 3, 3, device=q.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R

# Example usage
q = torch.randn(2, 4)  # Random quaternion (B, 4)
q = F.normalize(q, p=2, dim=-1)  # Normalize

R = quaternion_to_rotation_matrix(q)  # (B, 3, 3)
print(R.shape)  # (2, 3, 3)
