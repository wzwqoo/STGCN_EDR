## APPENDIX B: UAV SWARM CONFIGURATION AND SENSOR REQUIREMENTS

This appendix specifies the minimum and recommended UAV swarm architecture required to compute each feature in the proposed F = 28 input vector, the sensor suite each UAV must carry, the sampling requirements, and practical deployment guidance for each atmospheric layer. Requirements are expressed in terms of neighbours: same-altitude horizontal neighbours (H) and lower-altitude vertical neighbours (V).

### B.1 Neighbour Requirement Summary

*Table B.1. Neighbour requirements for each turbulence index. H = same-altitude horizontal neighbours; V = vertical column neighbours. 'Non-collin.' = non-collinear, i.e. not both on the same line through self.*

| Index | Unit | H nbrs | V nbrs | Notes |
|---|---|---|---|---|
| EI | ×10⁻⁷ s⁻² | 2 (non-collin.) | 1 below | Solve 2×2 for ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y |
| Ri | – | 0 | 1 below | Vertical shear + potential temperature gradient |
| N² | rad² s⁻² | 0 | 1 below | Numerator of Ri; same pair |
| Brown (BI) | m² s⁻² | 0 | 1 below | Requires S² and Ri from same vertical pair |
| CP | m² s⁻² | 0 | 1 below | Requires S² and Ri from same vertical pair |
| TKE | J kg⁻¹ | 0 | 0 | Own time-series at ≥5 Hz over ≥10 s window |
| Scorer l² | m⁻² | 0 | ≥3 levels | Needs d²U/dz²; 3-point second derivative |
| Dutton (Ei) | – | 2 (non-collin.) | 1 below | ∇T along wind + speed shear |
| CAPE / CIN | J kg⁻¹ | 0 | ≥5 (col.) | Column integral; more levels = more accurate |
| LI | K | 0 | to 500 hPa | Needs sounding to ≈5500 m ASL |
| SI | K | 0 | 850+500 hPa | Needs 850 hPa parcel source + 500 hPa target |
| K-Index | – | 0 | 850/700/500 | Three standard pressure levels required |
| TT | – | 0 | 850/500 hPa | Two pressure levels |
| SWEAT | – | 0 | 850/500 hPa | Also needs u, v wind at both levels |
| SRH | m² s⁻² | 0 | ≥4 (0–3 km) | Hodograph area; more levels = better |
| SCP | – | 0 | ≥5 (0–6 km) | Needs CAPE (deep) + SRH + BWD at 6 km |
| EHI | – | 0 | ≥5 (0–3 km) | Needs CAPE (deep) + SRH |
| PW | kg m⁻² | 0 | ≥2 (col.) | More levels improve integration accuracy |
| EDR | m²/³ s⁻¹ | 0 | 0 | Own time-series; ≥5 Hz; ≥10 s window |

### B.2 Recommended Altitude Layer Configuration
To simultaneously satisfy the requirements of all indices listed in Table B.1, the swarm should maintain UAVs at the following eight altitude levels. The approximate pressure levels are computed using the standard atmosphere relationship.

*Table B.2. Recommended altitude layers. AGL = above ground level. Pressure values are approximate and vary with temperature.*

| Layer | Alt. AGL (m) | Approx. p (hPa) | Platform type | Primary index contributions |
|---|---|---|---|---|
| 1 | 30–100 | ~1000 | Multi-rotor | TKE, EDR, near-surface Ri; ground-truth reference |
| 2 | 150–300 | ~980 | Multi-rotor | Surface Ri (pair with L1), low Scorer l² |
| 3 | 500–700 | ~940 | Multi-rotor / VTOL | EI horizontal stencil, Dutton speed shear |
| 4 | 1400–1600 | ~850 | Multi-rotor / VTOL | 850 hPa level for K, TT, SWEAT, SRH, SI parcel |
| 5 | 2900–3100 | ~700 | Fixed-wing / VTOL | 700 hPa level for K-Index mid-level dryness |
| 6 | 3500–4000 | ~640 | Fixed-wing | SRH upper boundary, BWD intermediate |
| 7 | 5300–5700 | ~500 | Fixed-wing | 500 hPa level for K, TT, LI, SI, SWEAT, SCP |
| 8 | 6000–6500 | ~470 | Fixed-wing | BWD upper level for SCP; SRH extended layer |

For a minimum viable deployment targeting only the shear-driven indices (EI, Ri, BI, CP) and TKE/EDR, layers 1–3 are sufficient. Adding layer 4 (850 hPa) enables K, TT, and basic SRH. Layers 5–8 are required for the full index suite including LI, SI, CAPE, SCP, and EHI.

### B.3 Horizontal Grid Configuration
Within each altitude layer, UAVs should be arranged in a two-dimensional grid or approximately regular pattern to provide the non-collinear horizontal pairs required for EI and Dutton. The minimum horizontal configuration is:

*Table B.3. Horizontal UAV count and coverage at each altitude layer.*

| Configuration | UAVs / layer | Coverage | Supported indices |
|---|---|---|---|
| Minimum (EI + Ri) | 3 | ~5 km | EI, Ri, BI, CP, Dutton, TKE, EDR |
| Recommended (full) | 9–20 | 25–50 km | All 19 indices + EDR |
| Production (dense) | 20–40 | 50–100 km | All indices with redundancy for fault tolerance |

The minimum configuration of 3 UAVs per layer provides UAV P (self) plus two non-collinear neighbours for the 2×2 horizontal gradient system. Placing the three UAVs at vertices of an equilateral triangle with side 5 km maximises the determinant of the geometry matrix M and minimises gradient estimation error. For a full 8-layer deployment with 9 UAVs per layer, the total fleet size is 72 actively flying platforms; a practical 3-shift rotation (1 flying, 1 charging, 1 standby) requires approximately 216 platforms for continuous operation.

### B.4 Required Sensor Suite Per UAV

*Table B.4. Minimum sensor suite for each UAV. All sensors should be rated for −40°C to +50°C and IP54 weather protection.*

| Sensor | Measured quantity | Min. spec | Used for |
|---|---|---|---|
| 3-axis anemometer | u, v, w | ±0.3 m/s, 10 Hz | EI, Ri, BI, CP, TKE, EDR, SRH, Dutton, Scorer |
| Thermometer (Pt100/NTC) | T | ±0.1 K, 1 Hz | θ, N², Ri, CAPE, LI, SI, K, TT, Scorer |
| Barometric pressure | p | ±1 Pa, 1 Hz | θ, altitude, all pressure-level indices |
| Chilled-mirror hygrometer | Td | ±0.3 K, 1 Hz | CAPE, CIN, LI, SI, K, TT, SWEAT, PW, EDR (e) |
| GNSS receiver | lat, lon, alt | ±1 m horiz., ±2 m vert. | Position for graph edges; Δz for shear indices |
| IMU (6-DOF) | attitude, accel. | 100 Hz | Wind correction (remove platform motion from u,v,w) |

The three-axis anemometer is the most critical sensor. For multi-rotor platforms, sonic anemometers mounted above the rotor plane on a rigid mast of ≥0.3 m provide the best compromise between rotor-wash rejection and structural practicality. Five-hole pressure probes integrated into the fuselage nose are preferred for fixed-wing platforms operating at altitudes > 1000 m. All wind measurements must be corrected for platform attitude using the co-located IMU before feature computation.

### B.5 Sampling Rate and Window Requirements

*Table B.5. Sampling rate and window requirements.*

| Quantity | Min. rate | Recommended | Rationale |
|---|---|---|---|
| Wind (u, v, w) — EDR/TKE | 5 Hz | 25 Hz | Resolve inertial subrange; 25 Hz gives lags 3–40 at 5 km/h |
| Wind (u, v) — shear indices | 0.1 Hz | 1 Hz | Shear indices need only mean wind over the flight leg |
| T, p, Td | 0.5 Hz | 1 Hz | Thermodynamic indices vary slowly; 1 Hz sufficient |
| GNSS position | 1 Hz | 10 Hz | 10 Hz needed for accurate Δz in rapidly climbing UAVs |
| EDR window length | 10 s | 30–60 s | Longer windows give more stable PSD and SF estimates |
| ST-GCN input window (T) | 10 steps | 10 steps | Fixed by model architecture; step = 10 s sensor cycle |

### B.6 Fault Tolerance and Degraded Operation
When individual UAVs are unavailable due to battery depletion, communication failure, or return-to-base events, the following degradation modes apply:

*Table B.6. Fault tolerance behaviour under common swarm failure scenarios. The Gaussian soft adjacency ensures that a UAV that drifts out of communication range loses influence gradually rather than abruptly.*

| Failure scenario | Affected indices | Degraded behaviour |
|---|---|---|
| One horizontal neighbour lost | EI, Dutton | Degrade to one-sided difference (~2× gradient error); EI still computable |
| Both horizontal neighbours lost | EI, Dutton | EI = 0 assumed (shear unknown); flag in output |
| Lower vertical neighbour lost | Ri, BI, CP, EI-VWS, Dutton-shear | All shear indices unavailable; ST-GCN uses self-loop only; log σ² increases |
| Entire altitude layer lost | K, TT, SWEAT, SRH, LI, SI | Pressure-level indices unavailable; ST-GCN message-passing compensates via remaining layers |
| UAV offline mid-swarm | All indices for that node | Soft adjacency weights for offline node approach 0; GCN aggregation unaffected |

```