# CHAPTER 4: EXPERIMENTAL DATA AND TRAINING FRAMEWORK

### 4.1 Public Data Sources 
To train, validate, and benchmark the Spatio-Temporal Graph Convolutional Network (ST-GCN) prior to live deployment, historical atmospheric data must be sourced from established meteorological archives. These repositories provide the foundational sensor readings (temperature, pressure, wind vectors) necessary to compute the 24 physics-informed turbulence indices.

Global Synoptic Data (ERA5): Maintained by the European Centre for Medium-Range Weather Forecasts (ECMWF) via the Copernicus Climate Data Store, the ERA5 reanalysis provides hourly, global atmospheric profiles. It is the primary source for computing broad-scale stability metrics like Convective Available Potential Energy (CAPE) and the K-Index.

High-Resolution Regional Data (HRRR): Operated by the National Oceanic and Atmospheric Administration (NOAA), the High-Resolution Rapid Refresh model provides 3-kilometer grid spacing updated hourly. This is essential for sourcing high-frequency localized wind shear and thermodynamic data.

In-Situ EDR Ground Truth (MADIS / AMDAR): NOAA’s Meteorological Assimilation Data Ingest System aggregates Aircraft Meteorological Data Relay (AMDAR) reports from commercial airline fleets. This archive provides real-world, historically recorded Eddy Dissipation Rate (EDR) measurements, serving as the "ground truth" target for training the model's uncertainty-aware loss function.

---

### 4.2 Critical Observational Gaps in Public Data

The development of a robust ST-GCN for atmospheric sensing faces three primary data challenges: spatial, temporal, and observational. Addressing these gaps is critical for ensuring the model generalizes across diverse meteorological regimes.

1. Spatiotemporal Grid and Location Misalignment:
There is a fundamental mismatch between the resolution of global meteorological models and the operational scale of UAV swarms. Standard reanalysis datasets, such as ERA5, provide data on a coarse 31 km grid, while high-resolution models like HRRR are limited to 3 km.

2. Temporal Misalignment and Latency:
Even when spatial coordinates overlap, a critical "sync gap" exists between authoritative meteorological data and high-frequency UAV observations. This makes it mathematically difficult to pair a specific sensor reading with a reliable training label. 

3. Absence of Direct EDR Observations:
The most significant hurdle in training turbulence models is that Eddy Dissipation Rate (EDR) is almost never measured directly; it is a derived statistical value. Direct EDR measurements are typically restricted to commercial aircraft at high altitudes (above 20,000 ft). There is a total absence of standardized EDR "ground truth" for the lower boundary layer where UAVs operate. Pilot Reports (PIREPs) are too subjective to serve as machine learning labels, as a "bump" is felt differently by different aircraft types.

### 4.3 From Centralised Pre-Training to Federated Fine-Tuning
The centralized proxy pre-training stage is feasible only as an initial baseline, as it relies on publicly accessible ERA5 reanalysis and non-restricted AMDAR observations to develop a foundational global model. However, this approach becomes untenable for high-precision operations due to critical spatiotemporal and observational misalignments.

First, there is a severe grid misalignment; global models like ERA5 operate on a coarse 31 km grid that "smooths out" the micro-scale eddies and localized gradients that a UAV swarm actually experiences. Second, a temporal misalignment exists because hourly model averages cannot be reliably synchronized with high-frequency UAV observations; a single hourly "snapshot" fails to capture the transient, non-linear bursts of turbulence occurring at the sub-second level. Finally, the absence of direct EDR observations in the lower boundary layer means there are no "ground truth" labels available in public datasets for UAV altitudes.

This is the fundamental motivation for the federated learning architecture presented in Section 4.4. Rather than transmitting raw observations, each operator's swarm trains a local copy of the global model on its own matched pairs using the FedProx objective, then transmits only the resulting weight deltas — the difference between the locally updated weights and the global model received at the start of the round. The central aggregation server combines these deltas into an improved global model using sample-weighted FedAvg, and the cycle repeats. The raw atmospheric observations never leave each operator's environment; only anonymous numerical vectors of model parameter changes are exchanged. 

The federated training framework, its FedProx loss function, aggregation protocol, and the role of predicted uncertainty in coordinating swarm behaviour, are described in detail in the following section.


### 4.4 Federated Training with FedProx
Standard FedAvg is known to suffer from client drift when clients train for many local steps on heterogeneous data — different weather regimes produce gradient updates that point in opposing directions in parameter space, causing the aggregated global model to underperform any individual client (Li et al., 2020). We address this with the FedProx regularisation term, which anchors each client's local weights to the current global model:

```text
L_i(w) = L_NLL(w; D_i) + (μ_prox/2) · ||w - w_global||²
```

where D_i is the local dataset of client i, μ_prox ∈ {0.001, 0.01, 0.1} is a hyperparameter controlling the proximal strength, and w_global is a frozen copy of the global model weights taken at the start of each federation round. The gradient of the proximal term pulls the local model back toward the global model whenever the NLL gradient would push it far into client-specific parameter regions.

Global aggregation after each round uses sample-count weighted FedAvg:

```text
w_global ← w_global + Σ_i (n_i / N_total) · (w_i - w_global)
```

where n_i is the number of training samples contributed by client i in the current round and N_total = Σ_i n_i. Clients with more turbulence events — which tend to be swarms operating in more meteorologically active regions — therefore contribute proportionally more to the global model update. A federation round is triggered when at least half the registered clients have check in within the round window, which occurs every 6–12 hours when swarms return to base or gain adequate link bandwidth.

### 4.5 Edge Deployment and Inference Latency
The ST-GCN model is designed for deployment on UAV-mounted edge computers, with the Raspberry Pi 4B (4 GB) as the primary target platform. The complete forward pass — including the soft adjacency recomputation, temporal encoding, two GCN layers, and voxel MLP — completes within the 10-second UAV sensor cycle budget under all tested swarm configurations. Dynamic INT8 quantisation of the Linear layers via PyTorch's dynamic quantisation API reduces model size from 0.7 MB to approximately 0.3 MB and provides a 1.5–2.5× latency reduction on ARM Cortex-A72 cores with 8-bit SIMD support, yielding mean inference latencies of 3.2 s on the Raspberry Pi 4B and 1.4 s on the Raspberry Pi 5 for a 20-node swarm.

### 4.6 Storage Feasibility Analysis (72-Hour Window)
Workflow for 72-Hour Federated PredictionThe Federated Learning workflow must handle the transition from "Local Sensing" to "Global Forecasting":
Step 1: Local Feature Extraction: Each UAV calculates the 28 features (Group A + B). The EDR is calculated using the high-frequency 25 Hz burst, then discarded to save space.

Step 2: Temporal Stacking: The edge device stacks the Multi-Scale buffers. The input to the ST-GCN becomes a sequence: $X_{t-72h}, \dots, X_t$.

Step 3: FedProx Update: Every $N$ hours, the swarm coordinator aggregates local weights. The Proximal Term ($\mu$) is critical here; because weather patterns change over 72 hours, $\mu$ prevents the model from over-fitting to a single passing storm (local heterogeneity).

Step 4: Inference: The global model is pushed back to the UAVs. Even with 72 hours of history, the compressed feature vector remains small enough (~14 MB) to fit comfortably in the RAM of a Raspberry Pi 4B/5 or Jetson Nano.

### 4.7 Onboard Data Flow 
1. Collect raw sensor readings at native sampling rate.
2. Apply high-pass filter (0.5 Hz) to u, v, w for rotor correction.
3. Receive position and wind readings from vertical and horizontal neighbours via the swarm mesh network.
4. Compute all 19 turbulence indices using the appropriate stencil (Table B.1).
5. Estimate EDR via SF, PSD, and variance methods; take median.
6. Concatenate 9 sensor readings + 19 indices to form x_t ∈ R^28;
7. Update a rolling hourly moving average for all 28 features, maintaining a cumulative 72-hour historical trajectory in non-volatile storage.
8. Store the hourly-averaged feature vectors ($x_{t-72h \dots t}$) as inputs. For training labels, map the self-labeled EDR calculated at step 5 to a future-offset target index (1 to 72 hours ahead).
9. Every 12 hours or upon landing, initiate local training on the 72-hour look-back dataset $D_i$. The objective is to predict future turbulence states based on the stored historical averages:Minimize $\mathcal{L}_i(w) = \mathcal{L}_{NLL}(w; D_i) + \frac{\mu_{prox}}{2} \|w - w_{global}\|^2$
10. Compute the weight delta $\Delta w = w_{local} - w_{global}$. Transmit only the delta to the central coordinator to preserve privacy and minimize bandwidth.
111. A round is triggered only when at least 50% of the fleet has checked in. The central station calculates the new global model by applying a sample-count weight to each delta. 

### 4.8 Conclusion
Storage Impact: By reducing 25 Hz raw data to 1-hour moving averages for the 72-hour cache, the storage footprint per UAV drops from ~240 MB to < 15 MB. This allows for years of flight data to be stored on a standard 32GB microSD card.

Latency: Steps 1–6 (Feature Engineering) consume ~100 ms on a Raspberry Pi 5. The ST-GCN inference remains the bottleneck at ~2.0 s (INT8), allowing a comfortable margin within the 10-second operational cycle.