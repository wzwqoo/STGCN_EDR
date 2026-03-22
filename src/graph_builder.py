import torch


# ─────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────

def soft_adjacency(
        positions: torch.Tensor,  # [N, 3]  (x, y, z) metres, updated each step
        sigma_h_m: float = 5_000.0,  # horizontal Gaussian width [m]
        sigma_v_m: float = 300.0,  # vertical Gaussian width [m]
        eps: float = 1e-4,  # prune edges below this weight
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian kernel soft adjacency — dynamic, smooth, no hard threshold.

    w_ij = exp( -(dx^2 + dy^2)/(2*sigma_h^2)  -  dz^2/(2*sigma_v^2) )

    Values are in (0, 1].  Distant UAVs get near-zero weight naturally;
    no step-function discontinuity when UAVs cross a radius boundary.

    sigma_h = 5 km  => UAVs 5 km apart get w = exp(-0.5) = 0.61
                       UAVs 10 km apart get w = exp(-2.0) = 0.14
    sigma_v = 300 m => UAVs one altitude layer apart get w = 0.61

    Call once per forward pass with current positions.  O(N^2), ~0.1 ms
    for N=40 on CPU.
    """
    pos = positions.float()
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # unsqueeze add dimension to [1, N, 3] and [N, 1, 3], diff is [N, N, 3]

    w = torch.exp(
        - diff[..., 0] ** 2 / (2.0 * sigma_h_m ** 2)
        - diff[..., 1] ** 2 / (2.0 * sigma_h_m ** 2)
        - diff[..., 2] ** 2 / (2.0 * sigma_v_m ** 2)
    )  # [N, N] in (0, 1]

    w.fill_diagonal_(0.0)  # redundant
    mask = w > eps  # drop negligible edges for sparsity
    src, dst = mask.nonzero(as_tuple=True) # True outputs separate tensors
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = w[src, dst]
    return edge_index, edge_weight


def build_uav_graph(positions, h_radius=8_000.0, v_radius=700.0):
    """Backward-compat alias for soft_adjacency."""
    return soft_adjacency(positions)

