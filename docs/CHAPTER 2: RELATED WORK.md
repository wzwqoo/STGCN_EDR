## CHAPTER 2: RELATED WORK

### 2.1 Classical Turbulence Diagnostics and Operational Forecasting

#### 2.1.1 Physics-Based Turbulence Indices
1. Shear-Driven Stability and the Richardson Number
The Richardson Number (Ri), established by Richardson (1920) and refined with a 0.25 stability criterion by Miles and Howard (1964), remains the foundational measure of buoyancy stabilization versus wind shear destabilization. Its physical mechanism describes the threshold where kinetic energy from shear overcomes thermal stratification to initiate Kelvin-Helmholtz instability. Operationally, NWP implementations are limited by vertical grid spacing, which often fails to resolve the thin shear layers where Ri reaches critical values.

2. Kinematic Deformation and the Ellrod Index
The Ellrod Index (1992) complements Ri by focusing on the frontogenetic dynamics that concentrate shear along jet-stream flanks. It calculates the sum of vertical wind shear and horizontal deformation—incorporating both stretching and shearing—to diagnose upper-level CAT. While effective for synoptic-scale jet dynamics, the index frequently fails in regions where turbulence is driven by gravity wave breaking rather than pure shear (Sharman et al., 2006).

3. Shear-Weighting: Brown and Colson-Panofsky Indices
The Brown (1973) and Colson-Panofsky (1965) indices address a critical Ri weakness: the occurrence of near-critical stability in regions with negligible absolute shear. These metrics multiply the square of the vertical wind shear by a stability-dependent suppression factor, effectively weighting the instability signal by available kinetic energy. This formulation ensures that only high-energy unstable layers are flagged, a logic now standard in ensemble turbulence guidance products.

4. Convective Potential: CAPE, CIN, and Lifted Indices
Convective turbulence metrics, ranging from the K-Index (1960) to Convective Available Potential Energy (CAPE), measure the buoyant energy available to updrafts based on moisture and lapse rates. Parcel theory (Emanuel, 1994) provides the physical grounding for CAPE and Convective Inhibition (CIN), defining the energy required to overcome capping inversions. Computationally cheaper approximations like the Lifted (1956) and Showalter (1953) indices offer single-level alternatives for real-time severe weather diagnosis.

5. Rotational Dynamics: Helicity and Composite Parameters
The Storm-Relative Helicity (SRH) (1990) characterizes the rotational potential of the low-level wind profile, moving beyond simple vertical motion to capture organized convective structures. When combined with CAPE in parameters like the Supercell Composite (2003) and Energy-Helicity Index (1991), it provides a multi-variate diagnostic for severe convective turbulence. These composite metrics are essential for identifying environments where vertical shear and buoyancy interact to sustain turbulent rotating updrafts.

6. Orographic Forcing and the Scorer Parameter
The Scorer Parameter (1949) defines the vertical propagation of mountain waves by analyzing the curvature of the wind profile and the Brunt-Väisälä frequency. Its physical mechanism describes "trapped" wave energy; a sharp decrease in the parameter with altitude signifies that energy is ducted and eventually released through wave breaking. This produces severe low-level orographic turbulence in the lee of mountain ranges that remains invisible to standard shear-based indices (Durran, 1990).

#### 2.1.2 Operational Turbulence Forecasting Systems
The Graphical Turbulence Guidance (GTG) system (Sharman et al., 2006; Sharman and Pearson, 2017) integrates more than fifteen individual turbulence diagnostics — including Ellrod variants, the Brown Index, Richardson Number, and CAPE-based indices — through a Bayesian ensemble weighting scheme trained against pilot reports (PIREPs) and aircraft-measured EDR. GTG3, the current operational version, runs on the NOAA Rapid Refresh (RAP) model at 13 km horizontal resolution and provides probabilistic turbulence guidance at 1-hour intervals for the conterminous United States. Its successor GTG4 incorporates deep learning ensemble post-processing. Despite strong performance at upper levels, GTG has known skill deficiencies in the boundary layer and for mountain-wave turbulence below the tropopause (Kim and Chun, 2012).

Recent work has demonstrated that machine learning ensemble methods consistently outperform individual GTG diagnostics when trained on large PIREP and EDR datasets. Sharman and Pearson (2017) showed that Random Forest models trained on GTG diagnostic output achieved superior skill scores to any individual index. Kim et al. (2021) extended this to gradient boosting with SHAP-based feature attribution, finding that the Ellrod Index variants, the Richardson Number, and TKE from the boundary layer parameterisation scheme were the highest-ranked predictors for moderate-or-greater turbulence at cruise altitudes.

### 2.2 Machine Learning for Turbulence Prediction

#### 2.2.1 Tabular and Ensemble Methods
The most successful operational machine learning systems for turbulence prediction treat the problem as supervised regression or binary classification on tabular feature vectors derived from NWP model output. Gradient Boosted Decision Trees (GBDTs) — specifically XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017) — have demonstrated state-of-the-art performance on aircraft-reported EDR datasets, with area-under-curve values exceeding 0.85 for moderate-or-greater turbulence classification at 500 hPa (Sharman and Pearson, 2017). These methods are computationally efficient, interpretable through SHAP values, and robust to missing features — properties that make them practical for operational deployment.

However, tabular GBDT methods have a fundamental architectural limitation for the swarm-sensing application: they treat each observation as an independent row in a feature matrix, with no mechanism to represent the spatial relationships between simultaneously collected observations at different locations. In a UAV swarm, the relative position and wind direction between two UAVs — specifically whether one is upstream of the other — carries direct physical information about how turbulence detected at the upstream UAV will propagate to the downstream one. GBDTs can only incorporate this information through manually engineered pairwise features, a process that scales quadratically with swarm size and requires the spatial structure to be known at feature engineering time rather than learned from data.

#### 2.2.2 Deep Learning Approaches
Convolutional neural networks (CNNs) applied to gridded NWP model fields have been explored for turbulence prediction since approximately 2018. Lee and Chun (2021) trained a ResNet-based architecture on Global Forecast System (GFS) wind and temperature fields to predict the spatial distribution of CAT, achieving substantial improvements over GTG at upper levels. However, CNNs require data on regular grids, which is incompatible with the irregular spatial distribution of a mobile UAV swarm. Gridding UAV observations by interpolation introduces smoothing artefacts and discards the exact measurement positions that are physically meaningful.

Recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks have been applied to single-station turbulence time series prediction, capturing temporal autocorrelation in successive turbulence reports at fixed measurement points (Kim and Chun, 2016). While LSTMs model temporal dynamics effectively, they have no built-in mechanism for spatial message passing between simultaneously operating sensors. Transformer-based architectures have been applied to spatially distributed meteorological prediction through global attention over all sensor pairs, but the O(N²) attention cost and the absence of inductive spatial biases limit their applicability to small swarms.

### 2.3 Graph Neural Networks for Spatio-Temporal Atmospheric Modelling

#### 2.3.1 Graph Convolutional Networks
Kipf and Welling (2017) introduced the Graph Convolutional Network (GCN), formalising spectral graph convolution as a first-order approximation that can be computed efficiently using the normalised graph Laplacian. The resulting message-passing update rule — aggregating symmetrically normalised neighbour features — has become the canonical baseline for node-level prediction tasks on static graphs. Velickovic et al. (2018) extended this with Graph Attention Networks (GATs), which compute data-dependent edge weights through a learned attention mechanism, allowing the model to weight neighbour contributions by feature similarity rather than graph topology alone.

The application of GNNs to spatial sensor networks has seen rapid growth since 2020. Wu et al. (2019) proposed the WaveNet-based GWNET architecture for traffic flow prediction on road networks, demonstrating that adaptive adjacency matrices — learned end-to-end without relying on the physical road graph — substantially outperformed fixed-topology GNNs. This result is directly relevant to UAV swarms, where the physical proximity graph is an imperfect proxy for the true atmospheric information flow structure.

#### 2.3.2 Spatio-Temporal GNNs for Meteorological Applications
Spatio-Temporal GNNs combine graph convolution with temporal sequence modelling through one of three architectural patterns: sequential (GCN followed by RNN), parallel (GCN and temporal convolution in separate branches with feature concatenation), or interleaved (alternating spatial and temporal layers). The parallel pattern, exemplified by the ST-GCN architecture of Yan et al. (2018) and the STGCN of Yu et al. (2018), achieves the best balance of expressiveness and computational efficiency and is adopted in the present work.

For atmospheric applications, Lam et al. (2023) demonstrated that a graph transformer trained on ERA5 reanalysis data — treating each grid point as a graph node — achieved competitive medium-range weather forecasting skill against operational NWP systems at a fraction of the computational cost. Bi et al. (2023) reported similar findings with the Pangu-Weather architecture. These results establish that graph-based architectures can capture the non-local atmospheric teleconnections that govern large-scale weather evolution. However, neither system addresses the boundary-layer turbulence prediction problem or incorporates dynamic graph topology, as both operate on fixed regular grids derived from reanalysis products.

The closest antecedent to the proposed system is the work of Zhao et al. (2022), who applied a Dynamic Spatio-Temporal GNN to atmospheric PM2.5 prediction from a network of fixed air quality monitoring stations, demonstrating that adaptive edge weights derived from wind direction improved prediction accuracy over fixed distance-based graphs. Their adjacency matrix, however, was updated only at hourly intervals corresponding to NWP wind field updates, rather than continuously as required for a mobile UAV swarm. Ni et al. (2023) addressed mobile sensor network prediction with a dynamic graph construction scheme but did not incorporate uncertainty quantification or federated training.

### 2.4 UAV-Based Atmospheric Sensing

#### 2.4.1 Wind Measurement from UAV Platforms
Atmospheric measurement from UAV platforms has advanced substantially since the pioneering fixed-wing soundings of van den Kroonenberg et al. (2012). Multi-rotor platforms offer the advantage of controlled hovering flight, enabling stationary atmospheric profiling, but introduce rotor-wash contamination into vertical wind measurements that must be corrected through high-pass filtering (Palomaki et al., 2017) or momentum-balance estimation (Thielicke et al., 2021). Five-hole probe sensors mounted on fixed-wing platforms (Wildmann et al., 2014; Calmer et al., 2018) achieve wind measurement accuracy of 0.1–0.3 m/s at sampling rates of 100 Hz, sufficient to resolve turbulent eddies in the inertial subrange. Sonic anemometers mounted above the rotor plane of multi-rotor UAVs have demonstrated comparable accuracy with appropriate correction algorithms (Prudden et al., 2016).

Eddy Dissipation Rate estimation from UAV wind time series has been validated against tower-mounted sonic anemometers and tethered balloon soundings. The structure function method (Mann et al., 2010) and the power spectral density method (Siebert et al., 2006) have both been applied to UAV data, with agreement to within 20% of reference measurements when the Taylor frozen turbulence hypothesis is satisfied — that is, when the UAV airspeed is substantially greater than the turbulent velocity fluctuations (Lundquist and Bariteau, 2015). The variance method provides a faster but less accurate alternative useful for real-time applications where the sample window is too short for spectral analysis (Lenschow et al., 1994).

#### 2.4.2 UAV Swarm Sensing Architectures
Cooperative UAV swarm sensing for atmospheric research has been demonstrated at small scale by the LAPSE-RATE campaign (de Boer et al., 2020), which deployed up to eight UAVs in coordinated patterns to characterise boundary-layer structure over heterogeneous terrain. The MUAC (Multi-UAV Atmospheric Campaign) demonstrated concurrent profiling at multiple altitudes using a mix of multi-rotor and fixed-wing platforms. These campaigns established the operational feasibility of swarm sensing but did not address the machine learning inference problem or the data governance challenges that arise when swarm data is distributed across multiple operators.

The application of graph neural networks to UAV swarm data is an emerging research direction. Existing work has focused on UAV trajectory coordination and collision avoidance (Zhou et al., 2022) rather than atmospheric sensing. To our knowledge, the present work is the first to model a UAV atmospheric sensing swarm as a dynamic weighted graph and to apply GNN message passing for turbulence field reconstruction.

### 2.5 Federated Learning for Distributed Sensor Networks

#### 2.5.1 Foundations of Federated Learning
Federated learning (FL) was introduced by McMahan et al. (2017) as a privacy-preserving distributed training paradigm in which N clients collaboratively train a global model without sharing raw data. Each client trains locally on its private dataset and uploads model weight updates — gradients or weight deltas — to a central server that aggregates them using the FedAvg algorithm: a sample-count weighted average of the client parameter vectors. The key privacy guarantee is that the server never observes individual training examples, only model statistics.

A fundamental limitation of standard FedAvg is client drift: when clients train for multiple local epochs on heterogeneous data distributions, their local optima diverge in parameter space, and their weight updates can partially cancel each other at aggregation, degrading global model accuracy below the single-client baseline. Li et al. (2020) proved theoretically and demonstrated empirically that client drift grows with the number of local epochs, the heterogeneity of client data distributions, and the learning rate. The FedProx algorithm (Li et al., 2020) addresses drift by adding a proximal term to each client's local objective, penalising the local model for deviating from the global model and providing a convergence guarantee for non-IID data distributions.

#### 2.5.2 Federated Learning in Environmental and Atmospheric Applications
The application of federated learning to environmental sensing networks is a recent development driven by the proliferation of distributed IoT sensor arrays and the associated data governance challenges. Liu et al. (2022) demonstrated federated training of air quality prediction models across geographically distributed monitoring networks in China, showing that federated models achieved within 3% of centralised training accuracy while maintaining data locality. Chen et al. (2023) applied federated graph neural networks to precipitation nowcasting across national weather services in Europe, finding that FedProx with μ_prox = 0.01 consistently outperformed FedAvg under the inter-country data heterogeneity characteristic of continental-scale weather systems.

For UAV applications specifically, federated learning has been explored in the context of surveillance and navigation but not atmospheric sensing. The challenge of dynamic graph topology — where the swarm configuration changes between federation rounds — has not been previously addressed in the FL literature. Our work contributes a solution in which the graph is reconstructed from current UAV positions at each forward pass, decoupling the model architecture from any fixed topology assumption and allowing the federally trained global model to be deployed in any swarm configuration without retraining.

### 2.6 Research Gap and Positioning
Table 2.1 summarises the capabilities of the most closely related systems relative to the present work across six criteria: dynamic graph topology, physics-informed features, uncertainty quantification, federated training, edge hardware deployment, and in-situ UAV data.

*Table 2.1. Comparison of related systems across key capability dimensions. ✓ = addressed; ○ = partially addressed; ✗ = not addressed.*

| System | Dynamic graph | Physics features | Uncertainty | Federated | Edge hardware | UAV data |
|---|---|---|---|---|---|---|
| GTG / GTG4 (Sharman 2006, 2017) | ✗ | ✓ | ○ | ✗ | ✗ | ✗ |
| GBDT-EDR (Kim et al. 2021) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| CNN-CAT (Lee & Chun 2021) | ✗ | ○ | ✗ | ✗ | ✗ | ✗ |
| Pangu / GraphCast (2023) | ✗ | ○ | ✗ | ✗ | ✗ | ✗ |
| Dynamic ST-GNN (Ni et al. 2023) | ○ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Fed-GNN-AQ (Chen et al. 2023) | ✗ | ○ | ○ | ✓ | ✗ | ✗ |
| **Proposed ST-GCN (this work)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

The comparison reveals that no existing system simultaneously addresses all six capabilities. GTG and GBDT-based systems provide the strongest physics-informed feature sets but operate on static NWP grids with no dynamic spatial structure, no uncertainty quantification, and no federated training. Graph neural network approaches for atmospheric modelling are static and centralised. The proposed ST-GCN is the first system to unify all six capabilities in a single deployable framework, enabled by the combination of Gaussian kernel soft adjacency for dynamic graph construction, heteroscedastic output heads for uncertainty quantification, FedProx for distributed optimisation, and INT8 quantisation for edge deployment.

---