# SPATIO-TEMPORAL GRAPH CONVOLUTIONAL NETWORK FOR CLEAR-AIR TURBULENCE PREDICTION USING FEDERATED UAV SWARM SENSING


---

## ABSTRACT

Clear-Air Turbulence (CAT) poses a significant hazard to aviation, yet Numerical Weather Prediction models lack the resolution to capture sub-grid eddy dynamics. We propose a Dynamic Spatio-Temporal Graph Convolutional Network (ST-GCN) that transforms cooperative UAV swarms into a high-resolution atmospheric sensing grid. By implementing dynamic soft adjacency, our model maintains predictive continuity across an evolving voxel-resolution volume. To ensure flight safety, we integrate Predictive Uncertainty Quantification via a dual-head architecture trained on Gaussian Negative Log-Likelihood, allowing the model to self-calibrate without labeled variance data. High-fidelity training targets are generated on edge hardware using a robust median of three independent EDR estimation methods. To address data privacy and institutional silos, we employ a Federated Learning protocol using FedProx, mitigating client drift in meteorologically heterogeneous environments. This framework provides a scalable, private, and uncertainty-aware solution for real-time, fine-grained turbulence forecasting.


**Keywords:** Clear-Air Turbulence, Eddy Dissipation Rate, Spatio-Temporal Graph Convolutional Network, UAV Swarm, Federated Learning, FedProx, Uncertainty Quantification, Edge Computing

---

The project follows a modular pipeline: 
1. feature_engineering.py
Purpose: Transforms raw high-frequency sensor data into physics-informed features. 
2. graph_builder.py
Purpose: Constructs the swarm topology dynamically. 
3. st_gcn_cat.py
Purpose: The core neural architecture. 
4. EDR_labeling.py
Purpose: EDR Estimation: Generates self-labeled training targets by fusing the Structure Function (SF), Power Spectral Density (PSD), and Variance methods.
5. inference_benchmark.py
Purpose: Hardware validation for deployment. Edge Simulation: Profiles latency and memory usage on Raspberry Pi 4B/5 and Jetson Nano.








