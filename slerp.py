import torch

def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.
    q1, q2: (B, 4) batch of quaternions (w, x, y, z).
    t: Interpolation factor (0 to 1).
    """
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)  # Normalize
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)  # Normalize

    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)

    if torch.abs(dot_product).item() > 0.9995:  # If too close, use LERP
        return (1 - t) * q1 + t * q2

    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)

    q_interp = (torch.sin((1 - t) * theta) / sin_theta) * q1 + \
               (torch.sin(t * theta) / sin_theta) * q2

    return q_interp / torch.norm(q_interp, dim=-1, keepdim=True)  # Ensure unit quaternion

# Example Usage
q_start = torch.tensor([[1, 0, 0, 0]])  # Identity quaternion (no rotation)
q_target = torch.tensor([[0, 1, 0, 0]])  # 180-degree rotation around X-axis

t_values = torch.linspace(0, 1, 5)  # Interpolation steps (0% to 100%)

for t in t_values:
    q_interp = slerp(q_start, q_target, t)
    print(f"t={t:.2f}, Quaternion: {q_interp.numpy()}")
