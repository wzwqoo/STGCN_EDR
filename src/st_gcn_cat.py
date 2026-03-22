"""
Spatio-Temporal GCN for Clear-Air Turbulence voxel prediction.

Input features (per UAV, per timestep)
───────────────────────────────────────
  Group A — raw sensor readings        (9 features)
    u_ms, v_ms, w_ms                   wind components        [m/s]
    T_K, p_Pa, Td_K                    temp / pressure / dew  [K, Pa, K]
    alt_m, lat_norm, lon_norm          position (normalised)

  Group B — turbulence indices         (19 features)
    EI        Ellrod Index             [×10⁻⁷ s⁻²]
    Ri        Richardson Number        [dimensionless]
    Brown     Brown Index              [m² s⁻²]
    N2        Brunt-Väisälä            [rad s⁻¹]
    CP        Colson-Panofsky          [m² s⁻²]
    TKE       Turbulent KE             [J/kg]
    CAPE      Conv. Avail. PE          [J/kg]
    CIN       Conv. Inhibition         [J/kg]
    LI        Lifted Index             [K]
    SI        Showalter Index          [K]
    K         K-Index                  [–]
    TT        Total Totals             [–]
    SWEAT     SWEAT Index              [-]
    SRH       Storm-Rel. Helicity      [m² s⁻²]
    SCP       Supercell Composite      [-]
    EHI       Energy-Helicity Index    [-]
    PW        Precipitable Water       [kg/m²]
    Scorer    Scorer Parameter l²      [m⁻²]
    Dutton    Dutton Index             [–]

  Total F = 9 + 19 = 28 features per timestep per UAV

Output
──────
  Per-UAV local voxel patch of EDR (eddy dissipation rate) [m^(2/3) s^-1]
  Shape [Vx, Vy, Vz] = [3, 3, 3]  — a 3×3×3 neighbourhood centred on the UAV
  Flattened to [27] in the MLP, reshaped back to [3, 3, 3] for downstream use

  Interpretation:
    voxel[1,1,1] = EDR at the UAV's own location (centre voxel)
    voxel[i,j,k] = EDR prediction for the offset cell
                   (i-1)*dx, (j-1)*dy, (k-1)*dz from UAV position
    dx = dy = 5 km,  dz = 300 m  (matches typical swarm spacing)

Architecture
────────────
  Step 1  InputProjection     linear per-feature group embedding
  Step 2  TemporalEncoder     causal 1-D conv  →  [N, T, C]      z_i
  Step 3  Temporal pool       mean over T      →  [N, C]
  Step 4  GCN-1               message passing  →  [N, C]
  Step 5  GCN-2               message passing  →  [N, C]         h_i
  Step 6  Fusion              cat(pool(z_i), h_i) → [N, 2C]
  Step 7  VoxelMLP            2C → C → C → 27 → reshape [3,3,3]

Dimensions  F=28, T=10, C=64, Voxels=3×3×3=27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_builder import soft_adjacency

# ── Feature catalogue ─────────────────────────────────────────

SENSOR_FEATURES = [
    "u_ms", "v_ms", "w_ms",
    "T_K", "p_Pa", "Td_K",
    "alt_m", "lat_norm", "lon_norm",
]

INDEX_FEATURES = [
    "EI", "Ri", "Brown", "N2", "CP", "TKE",
    "CAPE", "CIN", "LI", "SI", "K",
    "TT", "SWEAT", "SRH", "SCP", "EHI", "PW",
    "Scorer", "Dutton",
]

ALL_FEATURES    = SENSOR_FEATURES + INDEX_FEATURES
F_SENSOR        = len(SENSOR_FEATURES)   # 9
F_INDEX         = len(INDEX_FEATURES)    # 19
F_TOTAL         = len(ALL_FEATURES)      # 28


VOXEL_SHAPE     = (3, 3, 3)             # Vx × Vy × Vz
VOXEL_FLAT      = 3 * 3 * 3            # 27
VOXEL_DX_M      = 5_000.0             # horizontal voxel spacing [m]
VOXEL_DZ_M      =   300.0             # vertical voxel spacing [m]


# ─────────────────────────────────────────────────────────────
# Step 1 — Input projection (sensor group + index group)
# ─────────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """
    Projects the two feature groups to a shared embedding space
    before the temporal encoder.

    Sensor features and index features are embedded separately
    (different linear layers) then added — this lets the model
    learn different scales/units for raw physics vs derived indices.

    Input  : [N, T, F]         N = 1 UAV, T = 10 timesteps, F = F_SENSOR + F_INDEX
    Output : [N, T, C]
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        C = hidden_dim
        self.sensor_proj = nn.Sequential(
            nn.Linear(F_SENSOR, C),
            nn.LayerNorm(C),
            nn.GELU(),
        )
        self.index_proj = nn.Sequential(
            nn.Linear(F_INDEX, C),
            nn.LayerNorm(C),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [N, T, F]
        x_sensor = x[..., :F_SENSOR]          # [N, T, 9]
        x_index  = x[..., F_SENSOR:]          # [N, T, 15]
        out = self.sensor_proj(x_sensor) + self.index_proj(x_index)
        return self.dropout(out)               # [N, T, C]


# ─────────────────────────────────────────────────────────────
# Step 2 — Causal temporal encoder
# ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Left-padded 1-D conv — output at t sees only inputs at ≤ t.
    Sliding window on last dimension T
    Input  : [N, C, T]
    Output : [N, C, T]
    """
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.pad  = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0))) # pad 2 to the left, 0 to the right


class TemporalEncoder(nn.Module):
    """
    Two stacked causal conv blocks.

    Block = CausalConv1d → BatchNorm → GELU → Dropout

    Input  : [N, T, C]
    Output : [N, T, C]
    """
    def __init__(self, hidden_dim: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        C = hidden_dim
        def _block():
            return nn.Sequential(
                CausalConv1d(C, kernel_size),
                nn.BatchNorm1d(C),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.block1 = _block()
        self.block2 = _block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [N, T, C]
        x_t = x.permute(0, 2, 1)       # [N, C, T]
        out = self.block1(x_t)
        out = self.block2(out)
        out = (out + x_t).permute(0, 2, 1)   # [N, T, C] residual connection
        return out


# ─────────────────────────────────────────────────────────────
# Steps 4–5 — Weighted GCN layer (pure PyTorch)
# ─────────────────────────────────────────────────────────────

class WeightedGCNConv(nn.Module):
    """
    Symmetric-normalised graph convolution with self-loops.

        m_i  = Σ_j  (e_ij / sqrt(d_i·d_j)) · h_j
        h_i' = GELU( LayerNorm( W·m_i + skip(h_i) ) )

    Input  : [N, C]
    Output : [N, C]
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear  = nn.Linear(dim, dim)
        self.norm    = nn.LayerNorm(dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor, # shape [2,E] [0, 1, 2], # Source nodes
                                                       #[1, 2, 3]  # Target nodes
                edge_weight: torch.Tensor) -> torch.Tensor:
        N   = h.size(0)
        dev = h.device
        dtype = h.dtype

        # add self-loops, append self index and weight = 1 at the end
        self_idx    = torch.arange(N, device=dev)
        ei          = torch.cat([edge_index, torch.stack([self_idx, self_idx])], dim=1)
        ew          = torch.cat([edge_weight.to(dtype), torch.ones(N, device=dev, dtype=dtype)])
        src, dst    = ei[0], ei[1]          # has dimension of E+N

        # degree
        deg = torch.zeros(N, device=dev, dtype=dtype)
        deg.scatter_add_(0, dst, ew)
        d   = deg.pow(-0.5).clamp(max=1e4)   # inv-sqrt degree, has dimension of N

        norm = d[src] * ew * d[dst]           # [E] norm has dimension of E+N

        # aggregate
        msg = h[src] * norm.unsqueeze(-1)   # h input [N, C], h[src] [E+N, C], unsqueeze has dimension [E+N, 1]
        agg = torch.zeros_like(h)           # agg output [N, C]
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg) # unsqueeze [E+N, 1], expand_as repeats C times to [E+N, C]

        out = self.linear(agg)
        out = self.norm(out + h)              # residual: add original h
        out = self.act(out)
        out = self.dropout(out)
        return out


# ─────────────────────────────────────────────────────────────
# Step 7 — Voxel MLP with uncertainty (mean + log-variance)
# ─────────────────────────────────────────────────────────────

class VoxelMLP(nn.Module):
    """
    Maps fused embedding to a 3x3x3 EDR voxel patch with uncertainty.

    Shared trunk: 2C -> C -> C
    Two heads from the trunk output:
        mu_head     : C -> 27   mean EDR per voxel     (Softplus, EDR >= 0)
        logvar_head : C -> 27   log variance per voxel (clamped to [-6, 4])

    The log-variance head learns uncertainty WITHOUT needing labelled
    variance data — the Gaussian NLL loss trains it jointly with mu.

    Input  : [N, 2C]
    Output : mu      [N, 3, 3, 3]   predicted EDR  [m^(2/3) s^-1]
             log_var [N, 3, 3, 3]   log sigma^2    (dimensionless)
    """
    # clamp log_var to keep variance in [exp(-6), exp(4)] = [0.0025, 54.6]
    LOG_VAR_MIN = -6.0
    LOG_VAR_MAX = 4.0

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        C = hidden_dim

        # shared trunk: 2C -> C -> C
        self.trunk = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.LayerNorm(C),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(C, C),
            nn.LayerNorm(C),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # mean head: C -> 27, EDR must be >= 0
        self.mu_head = nn.Linear(C, VOXEL_FLAT)
        self.mu_act = nn.Softplus(beta=5)

        # log-variance head: C -> 27, unconstrained then clamped
        self.logvar_head = nn.Linear(C, VOXEL_FLAT)

        # initialise log-var head bias to 0 (start with unit variance)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=0.1)

    def forward(self, x: torch.Tensor):
        # x : [N, 2C]
        shared = self.trunk(x)  # [N, C]

        mu = self.mu_act(self.mu_head(shared))  # [N, 27]  >= 0
        log_var = self.logvar_head(shared).clamp(  # [N, 27]
            self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        mu = mu.view(-1, *VOXEL_SHAPE)  # [N, 3, 3, 3]
        log_var = log_var.view(-1, *VOXEL_SHAPE)  # [N, 3, 3, 3]
        return mu, log_var


# ─────────────────────────────────────────────────────────────
# Gaussian NLL loss
# ─────────────────────────────────────────────────────────────

def gaussian_nll_loss(
        mu: torch.Tensor,  # [N, 3, 3, 3]  predicted mean
        log_var: torch.Tensor,  # [N, 3, 3, 3]  predicted log sigma^2
        target: torch.Tensor,  # [N, 3, 3, 3]  ground-truth EDR
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood loss (scalar).

        L = (1/2) * mean[ (y - mu)^2 / sigma^2  +  log sigma^2 ]

    where sigma^2 = exp(log_var).

    Properties
    ----------
    - Penalises large residuals (y - mu)^2 / sigma^2
    - Penalises predicting high variance when unnecessary (log sigma^2 term)
    - Automatically trained: model raises sigma^2 when it is uncertain,
      lowers it when confident — no variance labels needed.
    """
    # sigma^2 = exp(log_var)  but compute via log for numerical stability
    # L = 0.5 * [ (y-mu)^2 * exp(-log_var)  +  log_var ]
    residual_sq = (target - mu) ** 2  # [N, 3, 3, 3]
    loss = 0.5 * (residual_sq * torch.exp(-log_var) + log_var)
    return loss.mean()


# ─────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────

class STGCNTurbulence(nn.Module):
    """
    Spatio-Temporal GCN: sensor + met indices -> CAT voxel grid with uncertainty.

    Parameters
    ----------
    hidden_dim  : C, channel width throughout  (default 64)
    T           : causal window length          (default 10)
    kernel_size : causal conv kernel            (default 3)
    dropout     : dropout probability           (default 0.1)

    Input
    -----
    x            : [N, T, F_TOTAL=28]   sensor + index features
    edge_index   : [2, E]               graph edges (COO, both directions)
    edge_weight  : [E]                  1/distance weights

    Output
    ------
    mu      : [N, 3, 3, 3]   predicted EDR  [m^(2/3) s^-1]
    log_var : [N, 3, 3, 3]   log sigma^2 — confidence score
                             sigma = exp(0.5 * log_var)
                             high sigma  => model is uncertain
                             low  sigma  => model is confident
    """

    def __init__(
            self,
            hidden_dim: int = 64,
            T: int = 10,
            kernel_size: int = 3,
            dropout: float = 0.1,
    ):
        super().__init__()
        C = hidden_dim

        self.input_proj = InputProjection(C, dropout)
        self.temporal_enc = TemporalEncoder(C, kernel_size, dropout)
        self.gcn1 = WeightedGCNConv(C, dropout)
        self.gcn2 = WeightedGCNConv(C, dropout)
        self.voxel_mlp = VoxelMLP(C, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
            self,
            x: torch.Tensor,  # [N, T, F_TOTAL]
            positions: torch.Tensor,  # [N, 3]  current UAV positions [m]
            sigma_h_m: float = 5_000.0,  # soft_adjacency horizontal sigma
            sigma_v_m: float = 300.0,  # soft_adjacency vertical sigma
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (mu, log_var) each [N,3,3,3]
        """
        Parameters
        ----------
        x          : [N, T, F_TOTAL]  sensor + index features
        positions  : [N, 3]  current positions in metres — recomputed each call
                     so the graph always reflects the swarm's actual geometry.
        """
        # Recompute soft graph from current positions (Fix 2 — dynamic graph)
        edge_index, edge_weight = soft_adjacency(
            positions.to(x.device), sigma_h_m, sigma_v_m
        )

        # Step 1: project sensor + index features to shared space
        e = self.input_proj(x)  # [N, T, C]

        # Step 2: causal temporal encoding
        z = self.temporal_enc(e)  # [N, T, C]

        # Step 3: pool time dimension
        x_pool = z.mean(dim=1)  # [N, C]

        # Steps 4-5: spatial message passing with dynamic soft edges
        h = self.gcn1(x_pool, edge_index, edge_weight)  # [N, C]
        h = self.gcn2(h, edge_index, edge_weight)  # [N, C]

        # Step 6: fuse temporal + spatial
        fused = torch.cat([x_pool, h], dim=-1)  # [N, 2C]

        # Step 7: predict voxel mean + uncertainty
        mu, log_var = self.voxel_mlp(fused)  # [N, 3, 3, 3] each
        return mu, log_var


# ─────────────────────────────────────────────────────────────
# Synthetic dataset for training demo
# ─────────────────────────────────────────────────────────────

class SyntheticCATDataset(torch.utils.data.Dataset):
    """
    Generates random (x, target_voxels) pairs for training demonstration.

    In production replace with real UAV observations and EDR labels
    from PIREP reports or aircraft-mounted EDR sensors.

    Each sample is a full swarm snapshot:
        x       : [N, T, F_TOTAL]  sensor + index features
        target  : [N, 3, 3, 3]    ground-truth EDR per voxel
    """

    def __init__(self, n_samples: int, N: int, T: int = 10, seed: int = 0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, N, T, F_TOTAL, generator=rng)
        # EDR ground truth: positive values, sparse (most air is calm)
        # simulate with Gamma-like distribution via Softplus of noise
        raw = torch.randn(n_samples, N, *VOXEL_SHAPE, generator=rng) * 0.5
        self.target = torch.nn.functional.softplus(raw, beta=5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.target[idx]


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(
        model: STGCNTurbulence,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        dataset: torch.utils.data.Dataset,
        n_epochs: int = 50,
        batch_size: int = 4,  # number of swarm snapshots per batch
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_every: int = 10,
) -> list[dict]:
    """
    Train the ST-GCN with Gaussian NLL loss.

    Each batch contains `batch_size` independent swarm snapshots.
    The graph topology (edge_index, edge_weight) is fixed across batches
    — in production you would recompute it if UAVs move significantly.

    Returns
    -------
    history : list of dicts with 'epoch', 'train_loss', 'val_loss'
    """
    model = model.to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    # 80/20 train/val split
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # cosine annealing: smoothly decay lr to lr/10 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr / 10
    )

    history = []

    for epoch in range(1, n_epochs + 1):
        # ── training ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            # x_batch : [B, N, T, F],  y_batch : [B, N, 3, 3, 3]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)

            # merge batch and node dimensions for a single forward pass
            # [B, N, T, F] -> [B*N, T, F]
            x_flat = x_batch.view(B * x_batch.size(1), *x_batch.shape[2:])
            y_flat = y_batch.view(B * y_batch.size(1), *y_batch.shape[2:])

            # repeat graph for each item in batch
            # shift node indices by N*b for sample b
            N_nodes = x_batch.size(1)
            ei_batch_list, ew_batch_list = [], []
            for b in range(B):
                ei_batch_list.append(ei + b * N_nodes)
                ew_batch_list.append(ew)
            ei_batch = torch.cat(ei_batch_list, dim=1)
            ew_batch = torch.cat(ew_batch_list, dim=0)

            optimiser.zero_grad()
            mu, log_var = model(x_flat, ei_batch, ew_batch)
            loss = gaussian_nll_loss(mu, log_var, y_flat)
            loss.backward()
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validation ────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                B = x_batch.size(0)
                x_flat = x_batch.view(B * x_batch.size(1), *x_batch.shape[2:])
                y_flat = y_batch.view(B * y_batch.size(1), *y_batch.shape[2:])
                ei_batch_list, ew_batch_list = [], []
                for b in range(B):
                    ei_batch_list.append(ei + b * N_nodes)
                    ew_batch_list.append(ew)
                ei_batch = torch.cat(ei_batch_list, dim=1)
                ew_batch = torch.cat(ew_batch_list, dim=0)
                mu, log_var = model(x_flat, ei_batch, ew_batch)
                val_loss += gaussian_nll_loss(mu, log_var, y_flat).item()
        val_loss /= len(val_loader)

        scheduler.step()

        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(rec)

        if epoch % log_every == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  epoch {epoch:>4d}/{n_epochs}  "
                  f"train_nll={train_loss:.4f}  "
                  f"val_nll={val_loss:.4f}  "
                  f"lr={lr_now:.2e}")

    return history


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────


def federated_train(
        global_model: "STGCNTurbulence",
        client_datasets: list,  # list of (dataset, positions) per client
        n_rounds: int = 20,
        local_epochs: int = 5,
        local_lr: float = 5e-4,
        weight_decay: float = 1e-4,
        mu_prox: float = 0.01,  # FedProx proximal coefficient
        batch_size: int = 4,
        device: str = "cpu",
        log_every: int = 5,
) -> list[dict]:
    """
    Federated training with FedProx loss.

    Loss per client:
        L = gaussian_nll_loss(mu, log_var, y)
          + (mu_prox / 2) * sum(||w - w_global||^2)

    The proximal term prevents client drift — it penalises each client's
    weights from straying too far from the current global model.  Without
    it, clients that train many local epochs on region-specific data can
    diverge until their updates cancel each other out at aggregation.

    Algorithm (FedAvg + FedProx)
    ─────────────────────────────
      for each round r:
        broadcast global weights  w_global  to all clients
        for each client i in parallel:
            load local dataset  (x, y, positions)
            run local_epochs of gradient descent with FedProx loss
            compute delta_i = w_i - w_global
        aggregate:  w_global += sum_i( n_i / N_total * delta_i )

    Parameters
    ----------
    global_model    : shared STGCNTurbulence instance (mutated in-place)
    client_datasets : list of (CATDataset, positions_tensor [N,3])
                      one entry per federated client (UAV swarm region)
    n_rounds        : number of federation rounds
    local_epochs    : gradient steps per round per client
    local_lr        : client learning rate
    mu_prox         : FedProx proximal strength (0 = standard FedAvg)
                      typical range 0.001 – 0.1; larger = less drift
    batch_size      : samples per local gradient step
    device          : "cpu" or "cuda"
    log_every       : print round summary every N rounds

    Returns
    -------
    history : list of dicts  {round, clients, avg_local_loss, n_samples_total}
    """
    global_model = global_model.to(device)
    history = []
    n_clients = len(client_datasets)

    def _proximal_term(model, w_global_flat):
        """||w_local - w_global||^2 summed over all parameters."""
        w_local_flat = torch.cat([
            p.view(-1) for p in model.parameters()
        ])
        return torch.sum((w_local_flat - w_global_flat) ** 2)

    def _flatten_weights(model):
        return torch.cat([p.detach().view(-1) for p in model.parameters()])

    def _load_weights(model, flat):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view(p.shape))
            idx += n

    for rnd in range(1, n_rounds + 1):

        # snapshot current global weights (used for delta and proximal term)
        w_global_flat = _flatten_weights(global_model).to(device)

        client_deltas = []  # list of (delta_flat, n_samples)
        round_losses = []

        for client_id, (dataset, positions) in enumerate(client_datasets):

            # ── initialise client model from global weights ──
            import copy
            client_model = copy.deepcopy(global_model)
            client_model.train()
            positions_dev = positions.to(device)

            optimiser = torch.optim.AdamW(
                client_model.parameters(),
                lr=local_lr,
                weight_decay=weight_decay,
            )

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )

            epoch_loss = 0.0
            n_batches = 0

            for _ in range(local_epochs):
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    B = x_batch.size(0)
                    N_nodes = x_batch.size(1)

                    # flatten batch dimension
                    x_flat = x_batch.view(B * N_nodes, *x_batch.shape[2:])
                    y_flat = y_batch.view(B * N_nodes, *y_batch.shape[2:])

                    # tile positions for the batch
                    pos_batch = positions_dev.unsqueeze(0).expand(
                        B, -1, -1
                    ).reshape(B * N_nodes, 3)

                    # forward — soft_adjacency recomputed inside model.forward
                    # but positions_dev is per-swarm; tile per sample in batch
                    # (all samples in one batch share the same swarm positions)
                    mu, log_var = client_model(x_flat, positions_dev)

                    # ── FedProx loss ──
                    nll = gaussian_nll_loss(mu, log_var, y_flat)
                    prox = (mu_prox / 2.0) * _proximal_term(
                        client_model, w_global_flat
                    )
                    loss = nll + prox

                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        client_model.parameters(), max_norm=1.0
                    )
                    optimiser.step()

                    epoch_loss += nll.item()  # log NLL only (not prox)
                    n_batches += 1

            # ── compute delta: local weights minus global weights ──
            w_local_flat = _flatten_weights(client_model).to(device)
            delta = w_local_flat - w_global_flat
            n_samples = len(dataset)

            client_deltas.append((delta, n_samples))
            avg_loss = epoch_loss / max(n_batches, 1)
            round_losses.append(avg_loss)

        # ── FedAvg aggregation: weighted sum of deltas ──
        total_samples = sum(n for _, n in client_deltas)
        agg_delta = torch.zeros_like(w_global_flat)
        for delta, n in client_deltas:
            agg_delta += (n / total_samples) * delta

        # update global model in-place
        w_new = w_global_flat + agg_delta
        _load_weights(global_model, w_new)

        avg_round_loss = sum(round_losses) / len(round_losses)
        rec = {
            "round": rnd,
            "clients": n_clients,
            "avg_local_nll": avg_round_loss,
            "n_samples_total": total_samples,
            "mu_prox": mu_prox,
        }
        history.append(rec)

        if rnd % log_every == 0 or rnd == 1:
            print(f"  round {rnd:>3d}/{n_rounds}  "
                  f"clients={n_clients}  "
                  f"avg_nll={avg_round_loss:.4f}  "
                  f"mu_prox={mu_prox}  "
                  f"samples={total_samples}")

    return history


if __name__ == "__main__":
    import time

    torch.manual_seed(0)

    C = 64
    T = 10
    N_GRID_X = 5
    N_GRID_Y = 4
    N_ALT = 2

    positions = torch.tensor([
        [ix * 5000.0, iy * 5000.0, iz * 600.0 + 150.0]
        for ix in range(N_GRID_X)
        for iy in range(N_GRID_Y)
        for iz in range(N_ALT)
    ])
    N = positions.shape[0]

    model = STGCNTurbulence(hidden_dim=C, T=T)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # soft adjacency — no hard radius, smooth decay
    ei, ew = soft_adjacency(positions)

    print("=" * 60)
    print("Model summary")
    print("=" * 60)
    print(f"  UAVs (N):            {N}  ({N_GRID_X}x{N_GRID_Y} x {N_ALT} alt)")
    print(f"  Input features (F):  {F_TOTAL}  = {F_SENSOR} sensor + {F_INDEX} indices")
    print(f"  Graph:               soft adjacency (Gaussian kernel, no hard threshold)")
    print(f"  Edges at t=0:        {ei.shape[1]}  (eps=1e-4 sparsity threshold)")
    print(f"  Parameters:          {n_params:,}")

    # ── single forward pass with dynamic positions ──
    x = torch.randn(N, T, F_TOTAL)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        mu, log_var = model(x, positions)  # positions passed directly
        t_fwd = time.perf_counter() - t0

    sigma = torch.exp(0.5 * log_var)
    print(f"\n  Forward pass:  {t_fwd * 1000:.2f} ms  (includes soft_adjacency recompute)")
    print(f"  mu shape:      {tuple(mu.shape)}")

    uav = 0
    print(f"\n── UAV {uav} centre voxel ──")
    print(f"  EDR mu    = {mu[uav, 1, 1, 1].item():.4f} m^(2/3) s^-1")
    print(
        f"  sigma     = {sigma[uav, 1, 1, 1].item():.4f}  ({'uncertain' if log_var[uav, 1, 1, 1] > 0 else 'confident'})")
    print(f"  95%% CI  = [{max(0, mu[uav, 1, 1, 1].item() - 1.96 * sigma[uav, 1, 1, 1].item()):.4f}, "
          f"{mu[uav, 1, 1, 1].item() + 1.96 * sigma[uav, 1, 1, 1].item():.4f}]")

    # ── demonstrate dynamic graph: UAV 0 drifts 9 km east ──
    print("\n── Dynamic graph demo: UAV 0 drifts 9 km east ──")
    pos_t0 = positions.clone()
    pos_t1 = positions.clone()
    pos_t1[0, 0] += 9000.0  # UAV 0 moves 9 km east
    ei0, ew0 = soft_adjacency(pos_t0)
    ei1, ew1 = soft_adjacency(pos_t1)
    # find edge 0->1 weight at t=0 and t=1
    mask0 = (ei0[0] == 0) & (ei0[1] == 1)
    mask1 = (ei1[0] == 0) & (ei1[1] == 1)
    w0 = ew0[mask0].item() if mask0.any() else 0.0
    w1 = ew1[mask1].item() if mask1.any() else 0.0
    print(f"  Edge 0->1 weight at t=0 (dist=5km):  {w0:.4f}")
    print(f"  Edge 0->1 weight at t=1 (dist=~10km): {w1:.4f}  (smooth decay, not a step)")
    print(f"  Edges at t=0: {ei0.shape[1]}   Edges at t=1: {ei1.shape[1]}")

    # ── centralized training ──
    print("\n" + "=" * 60)
    print("Centralised training (30 epochs)")
    print("=" * 60)
    dataset = SyntheticCATDataset(n_samples=60, N=N, T=T)
    history = train(
        model, ei, ew, dataset,
        n_epochs=30, batch_size=4, lr=1e-3, log_every=10,
    )
    print(f"  Final val NLL: {history[-1]['val_loss']:.4f}")

    # ── federated training ──
    print("\n" + "=" * 60)
    print("Federated training  (FedProx, 4 clients, 10 rounds)")
    print("=" * 60)

    # 4 clients = 4 regional swarms, each with its own positions
    # In production each client would be a separate UAV swarm in a different
    # geographic region; here we simulate with small position offsets
    client_configs = [
        (SyntheticCATDataset(n_samples=30, N=N, T=T, seed=i),
         positions + torch.tensor([i * 50_000.0, 0.0, 0.0]))
        for i in range(4)
    ]

    import copy

    global_model = copy.deepcopy(model)

    fed_history = federated_train(
        global_model=global_model,
        client_datasets=client_configs,
        n_rounds=10,
        local_epochs=3,
        local_lr=5e-4,
        mu_prox=0.01,
        batch_size=4,
        log_every=5,
    )

    print(f"\n  Final round avg NLL:  {fed_history[-1]['avg_local_nll']:.4f}")
    print(f"  Total samples used:   {fed_history[-1]['n_samples_total']}")
    print(f"  mu_prox (FedProx):    {fed_history[-1]['mu_prox']}")
    print("\n  What mu_prox does:")
    print("    0.0   = FedAvg (no proximal term, may diverge with long local training)")
    print("    0.01  = light anchoring (recommended for moderate drift)")
    print("    0.1   = strong anchoring (use when clients are very heterogeneous)")
