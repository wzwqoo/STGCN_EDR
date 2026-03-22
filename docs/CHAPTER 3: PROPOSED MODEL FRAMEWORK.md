## CHAPTER 3: PROPOSED MODEL FRAMEWORK
### 3.1 System Overview
We propose a Spatio-Temporal Graph Convolutional Network (ST-GCN) for real-time Clear-Air Turbulence (CAT) prediction from a distributed Unmanned Aerial Vehicle (UAV) swarm. Each UAV acts as a mobile atmospheric sensor, collecting multi-modal measurements at configurable sampling rates of 5–25 Hz. The swarm is modelled as a dynamic weighted graph G = (V, E, W), where each node v ∈ V represents one UAV, each edge (i, j) ∈ E encodes spatial proximity between UAVs i and j, and the edge weight w_ij reflects the Gaussian-decayed influence of one node's observations on another. The graph topology is recomputed at every inference step to accommodate the continuous positional drift inherent to autonomous aerial platforms.

As a demonstration, the model accepts a 10-timestep causal history of F = 28 features per UAV — comprising 9 raw sensor readings and 19 pre-computed meteorological turbulence indices — and produces per-UAV local voxel predictions of the Eddy Dissipation Rate (EDR), the ICAO-standard turbulence intensity metric measured in m²/³ s⁻¹. Each prediction is accompanied by a learned uncertainty estimate, enabling uncertainty-aware routing and alert decisions. Training across geographically distributed swarm operators is conducted via a federated learning protocol based on the FedProx algorithm, ensuring that raw sensor data never leaves each operator's local environment.

### 3.2 Input Feature Construction

#### 3.2.1 Raw Sensor Group (F_A = 9)
Each UAV is equipped with a three-axis sonic anemometer, a temperature and pressure sensor suite, and a dewpoint hygrometer. At each timestep t the following measurements are recorded: the zonal, meridional, and vertical wind components (u, v, w) in m s⁻¹; air temperature T in Kelvin; static pressure p in Pascals; dewpoint temperature T_d in Kelvin; and the normalised geodetic position (lat_norm, lon_norm, alt_m). Prior to model ingestion, all wind components are passed through a first-order high-pass Butterworth filter with a cut-off frequency of 0.5 Hz to remove low-frequency platform motion artefacts and rotor-induced downwash bias, a known source of systematic error in multi-rotor UAV anemometry (Prudden et al., 2016; Palomaki et al., 2017).

#### 3.2.2 Meteorological Index Group (F_B = 19)
Nineteen turbulence and convective instability indices are computed onboard each UAV at each sampling interval using measurements from the UAV itself and its immediate vertical and horizontal neighbours, as described in Section 2. These indices constitute the second feature group and span three physical mechanisms of atmospheric turbulence:

Shear-driven indices: the Ellrod Index (EI), Richardson Number (Ri), Brown Index (BI), and Colson-Panofsky Index (CP) quantify the competition between vertical wind shear and atmospheric stability that governs Kelvin-Helmholtz instability onset. Ri < 0.25 indicates turbulent onset (Miles and Howard, 1964).

Convective instability indices: CAPE, CIN, Lifted Index (LI), Showalter Index (SI), K-Index, and Total Totals (TT) characterise the thermodynamic environment governing deep moist convection, which is the dominant source of convective turbulence at lower tropospheric levels.

Synoptic-scale and moisture indices: Storm-Relative Helicity (SRH), Precipitable Water (PW), the Scorer Parameter l², and the Dutton Index represent large-scale dynamics, moisture loading, mountain-wave trapping potential, and cross-isobaric temperature gradients, respectively.

In addition, the Eddy Dissipation Rate (EDR) estimated from each UAV's own onboard wind time series — computed via the structure function method, the power spectral density method, and the variance method, with the median taken for robustness — is appended as a self-measured turbulence label that complements the model-predicted output.

*Table 1. Input feature groups.*

| Group | Count  | Features                                                                            | Physical role |
|---|--------|-------------------------------------------------------------------------------------|---|
| A — Sensor | 9      | u, v, w, T, p, Td, alt, lat, lon                                                    | Raw atmospheric state |
| B — Indices | 19     | EI, Ri, N2, BI, CP, TKE, CAPE, CIN, LI, SI, K, TT, SWEAT, SRH, SCP, EHI, PW, l², Ei | Physics-derived turbulence diagnostics |
| **Total** | **24** | **F_TOTAL = F_A + F_B**                                                             | **Per timestep, per UAV** |

### 3.3 Model Architecture
The proposed ST-GCN architecture processes the joint spatio-temporal input in seven sequential stages, each targeting a distinct aspect of the atmospheric turbulence prediction problem. The complete architecture contains approximately 180,000 trainable parameters with a hidden channel width C = 64 throughout, balancing expressiveness against the inference budget of edge devices such as the Raspberry Pi 4B.

#### 3.3.1 Input Projection (Step 1)
The two feature groups operate on fundamentally different numerical scales (raw pressure values reach ~10⁵ Pa while dimensionless indices such as Ri span [0, 1]), requiring separate embedding layers before their representations are combined. Group A and Group B are each independently projected by a Linear–LayerNorm–GELU sub-network into the shared hidden space R^C, after which their embeddings are element-wise added:

```text
e_t = W_A · x^A_t  +  W_B · x^B_t,      e_t ∈ R^C
```

where x^A_t ∈ R^9 and x^B_t ∈ R^19 are the sensor and index feature vectors at timestep t respectively, and W_A, W_B are learnable projection matrices. This design allows each group to learn an independent normalisation scale while sharing a common representational space for the downstream temporal encoder.

#### 3.3.2 Causal Temporal Encoder (Step 2)
Temporal dependencies are captured by two stacked causal one-dimensional convolutional blocks operating along the time axis. Causality is enforced by left-padding each sequence by (k - 1) positions prior to convolution, ensuring that the output at position t depends only on inputs at positions ≤ t and thus respects the chronological ordering of UAV observations. Each block applies a CausalConv1d layer followed by Batch Normalisation, GELU activation, and dropout:

```text
z^(l) = Dropout(GELU(BN(CausalConv1d(z^(l-1)))))
```

A residual connection from the projected input e to the output of the second block mitigates vanishing gradients and preserves low-frequency temporal trends. The encoder produces a temporal feature tensor Z ∈ R^(N×T×C), where N denotes the number of UAVs and T = 10 is the causal window length. The kernel size k = 3 was selected empirically to balance receptive field and computational cost on edge hardware.

#### 3.3.3 Temporal Pooling (Step 3)
The temporal dimension is compressed to a single spatial embedding by mean pooling across the T timesteps:

```text
x_pool = (1/T) Σ_{t=1}^{T} z_t,      x_pool ∈ R^(N×C)
```

Mean pooling was preferred over max pooling or attention-based temporal aggregation because atmospheric turbulence indices evolve slowly relative to the 10-step window, making the temporal mean a reliable summary statistic. The pooled representation x_pool serves as the node feature matrix for the subsequent graph convolutional layers.

#### 3.3.4 Dynamic Soft Adjacency (Graph Construction)
A key challenge in applying GNNs to UAV swarms is that the graph topology changes continuously as platforms manoeuvre. Hard-threshold adjacency matrices, where an edge is either present or absent depending on whether inter-UAV distance exceeds a fixed radius, introduce step-function discontinuities that destabilise training gradients whenever two UAVs cross the boundary distance. To address this, we replace the static hard-threshold graph with a differentiable Gaussian kernel soft adjacency:

```text
w_ij = exp( -(Δx²+Δy²)/(2σ_h²) - Δz²/(2σ_v²) )
```

where (Δx, Δy, Δz) = p_i - p_j is the relative position vector between UAVs i and j, σ_h = 5,000 m is the horizontal length scale, and σ_v = 300 m is the vertical length scale, each matching the typical inter-UAV spacing within a swarm layer. Edge weights decay smoothly from w = 1 at zero separation to w ≈ 0.61 at one standard deviation and w ≈ 0.14 at two standard deviations, with negligible edges (w < 10⁻⁴) pruned for computational efficiency. The soft adjacency is recomputed at every forward pass, ensuring that the graph always reflects the current geometric configuration of the swarm without any hard discontinuities.

#### 3.3.5 Graph Convolutional Layers (Steps 4–5)
Two successive weighted graph convolutional layers exchange compressed spatial embeddings between neighbouring UAVs. Each layer implements symmetric-normalised message passing with self-loops and a residual connection:

```text
m_i = Σ_{j∈N(i)} (w_ij / √(d_i · d_j)) · h_j
h_i' = GELU( LayerNorm( W · m_i + h_i ) )
```

where d_i = Σ_j w_ij is the weighted degree of node i and W ∈ R^(C×C) is a learnable weight matrix. Self-loops (w_ii = 1) ensure that each node's own representation is preserved through aggregation. The symmetric normalisation factor 1/√(d_i · d_j) follows Kipf and Welling (2017) and prevents the aggregated embeddings of high-degree nodes (those with many neighbours) from dominating the gradient signal. Two GCN layers allow each node's final representation to incorporate information from two-hop neighbours, covering a spatial radius of approximately 10–15 km under standard swarm configurations.

#### 3.3.6 Spatio-Temporal Fusion (Step 6)
The temporally-pooled representation x_pool (encoding local self-history) and the spatially-enriched representation h (encoding neighbourhood context) are concatenated to form the fused embedding:

```text
f_i =[x_pool_i || h_i],      f_i ∈ R^(2C)
```

This concatenation ensures that the MLP decoder receives both temporal context — what this UAV's own sensor history implies about its atmospheric state — and spatial context — what the surrounding swarm collectively reports. Ablation studies on synthetic data confirmed that omitting either branch degrades median EDR prediction error by 12–18%.

#### 3.3.7 Voxel MLP with Uncertainty Estimation (Step 7)
The fused embedding is decoded by a three-layer MLP into a 3×3×3 voxel patch of predicted EDR, centred on the UAV's current position with horizontal spacing Δx = Δy = 5 km and vertical spacing Δz = 300 m. The MLP employs a shared trunk followed by two parallel output heads:

```text
trunk(f_i) = LN(GELU(W_2 · LN(GELU(W_1 · f_i))))
μ_i = Softplus( W_μ · trunk(f_i) )        ∈ R^27
log σ²_i = clamp( W_σ · trunk(f_i), -6, 4 )  ∈ R^27
```

The mean head μ employs a Softplus activation to enforce the physical constraint EDR ≥ 0. The log-variance head log σ² is clamped to the interval [-6, 4], constraining the predictive variance to the range[e⁻⁶, e⁴] ≈ [0.0025, 54.6] m^(4/3) s⁻², and is initialised with near-zero weights to ensure the model begins training with unit variance before learning to calibrate uncertainty from data. Both 27-dimensional vectors are reshaped to[3, 3, 3] tensors. The centre voxel [1,1,1] corresponds to the EDR prediction at the UAV's own location.

### 3.4 Uncertainty-Aware Loss Function
The model is trained end-to-end with the Gaussian Negative Log-Likelihood (NLL) loss, which jointly optimises the predicted mean and variance without requiring labelled variance ground truth (Nix and Weigend, 1994; Kendall and Gal, 2017):

```text
L_NLL(θ) = (1/2N) Σ_i[ (y_i - μ_i)²/σ²_i + log σ²_i ]
```

where y_i ∈ R^27 is the ground-truth EDR voxel patch derived from PIREP reports or aircraft-mounted EDR sensors, μ_i is the predicted mean, and σ²_i = exp(log σ²_i) is the predicted variance. The loss has two complementary effects: the first term penalises large prediction errors, while the second term penalises over-confident predictions of high variance. Consequently, the model learns to output high σ when its prediction is uncertain — for instance, at the edges of the swarm where fewer neighbours contribute to the spatial aggregation — and low σ when it is confident. The resulting σ field provides operationally actionable confidence scores: cells with σ > 0.2 m^(1/3) s^(−1/2) are flagged for manual review before routing decisions are made.

### 3.5 Onboard EDR Estimation
Each UAV independently estimates its own EDR from its wind time series using three complementary methods before the ST-GCN forward pass. These self-estimated EDR values serve as ground-truth labels during online learning and as additional input features that enrich the model's self-awareness of local turbulence intensity.

The Structure Function method exploits Kolmogorov's (1941) inertial subrange scaling law: D_L(r) = C² · ε^(2/3) · r^(2/3), where D_L(r) is the second-order longitudinal structure function at spatial lag r = airspeed/f_s, C² = 2.0 is the Kolmogorov constant, and ε is the kinetic energy dissipation rate. EDR = ε^(1/3) is extracted by fitting the 2/3 power law to lags 3–40 samples.

The Power Spectral Density method fits a −5/3 slope to the log-log velocity PSD in the inertial subrange [0.5, f_s/4] Hz, using a pure-NumPy Welch estimator with Hanning windows. The EDR is recovered from the spectral intercept via the one-dimensional Kolmogorov constant A = 0.55.

The Variance method provides a fast approximate estimate: ε ≈ (σ_w / C_w)³ / L, where σ_w is the standard deviation of the high-pass filtered vertical wind, C_w = 1.9, and L = 100 m is an assumed outer length scale. This method is used when fewer than 50 samples are available.

The median of the three estimates is taken as the final reported EDR. A confidence score (high / medium / low) is assigned based on the inter-method spread and the goodness-of-fit of the spectral slopes relative to their theoretical Kolmogorov values.
