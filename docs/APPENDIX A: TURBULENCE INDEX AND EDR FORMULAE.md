## APPENDIX A: TURBULENCE INDEX AND EDR FORMULAE

This appendix provides the complete mathematical definitions of all turbulence indices and the Eddy Dissipation Rate (EDR) estimation methods used as input features and prediction targets in the proposed ST-GCN framework. All quantities are expressed in SI units unless stated otherwise. Subscripts i and j denote UAV node identifiers; subscripts low and high denote the lower and upper altitude levels of a vertical pair; subscripts 850, 700, 500 denote pressure levels in hPa.

### A.1 Common Symbols and Definitions

*Table A.1. Common symbols used throughout Appendix A.*

| Symbol | Unit | Definition |
|---|---|---|
| u, v | m s⁻¹ | Zonal (east) and meridional (north) wind components |
| w | m s⁻¹ | Vertical wind component (positive upward) |
| T | K | Absolute (dry-bulb) temperature |
| Td | K | Dewpoint temperature |
| p | Pa | Static pressure |
| θ | K | Potential temperature: θ = T(p₀/p)^κ, p₀ = 10⁵ Pa, κ = Rd/Cp ≈ 0.2854 |
| g | m s⁻² | Gravitational acceleration: g = 9.80665 m s⁻² |
| Rd, Rv | J kg⁻¹ K⁻¹ | Gas constants for dry air (287.05) and water vapour (461.5) |
| Cp | J kg⁻¹ K⁻¹ | Specific heat at constant pressure: Cp = 1004 J kg⁻¹ K⁻¹ |
| Lv | J kg⁻¹ | Latent heat of vaporisation: Lv = 2.501 × 10⁶ J kg⁻¹ |
| Δz | m | Altitude difference between upper and lower UAV: Δz = z_high - z_low |

### A.2 Shear-Driven Clear-Air Turbulence Indices

#### A.2.1 Ellrod Index (EI)
The Ellrod Turbulence Index (Ellrod and Knapp, 1992) diagnoses CAT by combining vertical wind shear with horizontal deformation of the wind field:

```text
EI = VWS × DEF        [× 10⁻⁷ s⁻²]
```

where the total deformation DEF and vertical wind shear VWS are defined as:

```text
DEF = sqrt(DST² + DSH²)
DST = ∂u/∂x − ∂v/∂y           (stretching deformation)
DSH = ∂v/∂x + ∂u/∂y           (shearing deformation)
VWS = sqrt((Δu/Δz)² + (Δv/Δz)²)
```

The horizontal partial derivatives ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y are estimated by solving the 2×2 linear system from two non-collinear same-altitude neighbours N₁ = (Δx₁, Δy₁, Δu₁, Δv₁) and N₂ = (Δx₂, Δy₂, Δu₂, Δv₂):

```text
[Δx₁  Δy₁] [∂u/∂x]   [Δu₁]
[Δx₂  Δy₂] [∂u/∂y] = [Δu₂]
```

provided det(M) = Δx₁Δy₂ − Δx₂Δy₁ ≠ 0 (neighbours not collinear with self). The same system is solved independently for the v-component. Output is scaled to conventional units of 10⁻⁷ s⁻². Severity thresholds: EI < 4 smooth; 4–8 light; 8–12 moderate; > 12 severe.

#### A.2.2 Richardson Number (Ri)
The bulk Richardson Number (Miles and Howard, 1964) measures the ratio of buoyant suppression to shear production of turbulence:

```text
Ri = N² / S²
N² = (g/θ_mean) × (Δθ/Δz)       (Brunt-Väisälä frequency squared)
S² = (Δu/Δz)² + (Δv/Δz)²        (wind shear squared)
θ_mean = (θ_high + θ_low) / 2
```

where Δθ = θ_high − θ_low, Δu = u_high − u_low, Δv = v_high − v_low. Ri is clipped to [−100, 100]. Critical thresholds: Ri < 0 convectively unstable; 0 ≤ Ri < 0.25 turbulent onset (KH instability); 0.25 ≤ Ri < 1 transitional; Ri ≥ 1 stable.

#### A.2.3 Brunt-Väisälä Frequency Squared (N²)
The Brunt-Väisälä frequency squared is the buoyancy restoring force per unit displacement in a stably stratified atmosphere:

```text
N² = (g / θ) × (∂θ/∂z)        [rad² s⁻²]
```

N² > 0 indicates stable stratification (gravity waves propagate); N² < 0 indicates convective instability. N² appears as the numerator of Ri and as the stability term in the Scorer Parameter.

#### A.2.4 Brown Index (BI)
The Brown Index (Brown, 1973) weights vertical wind shear squared by a stability factor to avoid false alarms from near-critical Ri at negligible shear magnitudes:

```text
BI = S² / (Ri + 1)            [m² s⁻²]
```

BI → VWS² as Ri → 0 (neutral); BI → 0 as Ri → ∞ (very stable). BI is undefined at Ri = −1; values are clamped to[0, ∞). Thresholds: < 2×10⁻⁴ smooth; 2–8×10⁻⁴ light; 8×10⁻⁴–2×10⁻³ moderate; > 2×10⁻³ severe.

#### A.2.5 Colson-Panofsky Index (CP)
The Colson-Panofsky Index (Colson and Panofsky, 1965) applies a linear stability suppression to shear, producing an exact zero at Ri = 1:

```text
CP = S² × (1 − Ri)            [m² s⁻²]
```

CP reaches its maximum at Ri = 0 (neutral atmosphere, maximum shear contribution) and is clamped to zero for Ri > 1. The (1 − Ri) factor is the fractional excess of shear production over buoyancy destruction in the turbulent kinetic energy budget. Thresholds: < 1.5×10⁻⁴ smooth; 1.5–6×10⁻⁴ light; 6×10⁻⁴–1.5×10⁻³ moderate; > 1.5×10⁻³ severe.

#### A.2.6 Dutton Index (Ei)
The Dutton Index (Dutton, 1969) combines speed shear and temperature gradient along the mean wind direction:

```text
Ei = 1.5 × |S_kt| + 0.5 × |ΔT_along|
```

where S_kt = speed shear in kt / 1000 ft and ΔT_along is the temperature difference in the wind direction extrapolated over 333 km (approx. 3° latitude). The temperature gradient ∇T = (∂T/∂x, ∂T/∂y) is computed from two non-collinear same-altitude neighbours via the same 2×2 system as EI, then projected onto the unit wind vector. Thresholds: Ei < 20 smooth; 20–30 light; 30–45 moderate; > 45 severe.

#### A.2.7 Turbulent Kinetic Energy (TKE)
TKE is computed directly from the Reynolds-decomposed wind time series sampled at ≥5 Hz over a ≥10 s window:

```text
TKE = ½ (σ²_u + σ²_v + σ²_w)         [J kg⁻¹]
```

where σ²_u = Var(u'), σ²_v = Var(v'), σ²_w = Var(w') are the variances of the high-pass filtered (cutoff 0.5 Hz) fluctuation components. Turbulence intensity TI = sqrt(2TKE/3) / U_mean is also reported. Thresholds: < 0.1 calm; 0.1–0.5 light; 0.5–2.0 moderate; > 2.0 severe.

#### A.2.8 Scorer Parameter (l²)
The Scorer Parameter (Scorer, 1949) governs the vertical propagation of mountain waves and their tendency to become trapped and break:

```text
l²(z) = N²(z)/U²(z) − (1/U(z)) × d²U/dz²      [m⁻²]
```

where U = sqrt(u² + v²) is the wind speed magnitude and d²U/dz² is its second vertical derivative, estimated by second-order centred differences from ≥3 altitude levels. l² > 0 with decreasing l² with height indicates wave trapping; l² < 0 indicates evanescent (non-propagating) waves. Thresholds: < 0 evanescent; 0–10⁻⁶ weak; 10⁻⁶–10⁻⁵ moderate; > 10⁻⁵ strong/trapping.

### A.3 Convective Instability Indices

#### A.3.1 Convective Available Potential Energy (CAPE)
CAPE quantifies the integrated positive buoyancy energy available to a rising parcel:

```text
CAPE = ∫_{z_LFC}^{z_EL} g × (T_parcel − T_env) / T_env dz     [J kg⁻¹]
```

The parcel is lifted from the surface dry-adiabatically to the Lifting Condensation Level (LCL) — height at which the parcel reaches saturation — then moist-adiabatically to the Equilibrium Level (EL). The LCL temperature is estimated via the Bolton (1980) formula: T_LCL = 1/(1/(Td − 56) + ln(T/Td)/800) + 56. The moist adiabatic lapse rate at each level is:

```text
Γs = g(1 + Lv·ws/(Rd·T)) / (Cp + Lv²·ws/(Rv·T²))
```

where ws is the saturation mixing ratio. Integration uses the trapezoid rule over UAV altitude levels. Thresholds: < 300 stable; 300–1000 marginal; 1000–2500 moderate; > 2500 extreme.

#### A.3.2 Convective Inhibition (CIN)
CIN is the integrated negative buoyancy energy the parcel must overcome before reaching the Level of Free Convection (LFC):

```text
CIN = ∫_{z_sfc}^{z_LFC} g × (T_parcel − T_env) / T_env dz[J kg⁻¹] (negative)
```

CIN is always ≤ 0. Thresholds: > −25 no cap; −50 to −25 weak; −200 to −50 moderate; < −200 strong cap.

#### A.3.3 Lifted Index (LI)
The Lifted Index is the temperature difference at 500 hPa between the environment and a parcel lifted from the surface:

```text
LI = T_env(500) − T_parcel(500)       [K]
```

Negative LI (parcel warmer than environment) indicates instability. Thresholds: > 2 stable; 0–2 slightly unstable; −3–0 moderately unstable; −6 to −3 very unstable; < −6 extremely unstable. Requires sounding to reach 500 hPa (≈ 5500 m ASL).

#### A.3.4 Showalter Index (SI)
The Showalter Index (Showalter, 1953) lifts the parcel from 850 hPa rather than the surface, reducing sensitivity to the superadiabatic surface layer:

```text
SI = T_env(500) − T_parcel_850→500     [K]
```

The 850 hPa parcel is lifted moist-adiabatically to 500 hPa. SI is more appropriate than LI for elevated convection and regions with strong afternoon surface heating. Thresholds: > 3 stable; 1–3 slightly unstable; −3–1 moderately unstable; −6 to −3 very unstable; < −6 extremely unstable.

#### A.3.5 K-Index (K)
The K-Index (George, 1960) quantifies thunderstorm potential from three pressure levels:

```text
K = (T₈₅₀ − T₅₀₀) + Td₈₅₀ − (T₇₀₀ − Td₇₀₀)
```

where T and Td are temperature and dewpoint in °C at the subscripted pressure level. The first term measures the 850–500 hPa lapse rate; Td₈₅₀ adds low-level moisture; the last term subtracts mid-level dryness. Thresholds: < 15 no convection; 15–25 isolated; 25–35 scattered; 35–40 numerous; > 40 extreme.

#### A.3.6 Total Totals Index (TT)
Total Totals (Miller, 1972) decomposes into Vertical Totals (VT) and Cross Totals (CT):

```text
TT = VT + CT
VT = T₈₅₀ − T₅₀₀                  (lapse rate component)
CT = Td₈₅₀ − T₅₀₀                 (moisture × cold-air component)
```

Thresholds: < 44 no convection; 44–50 isolated; 50–55 scattered; 55–60 severe; > 60 extreme.

#### A.3.7 SWEAT Index
The Severe Weather Threat Index (Miller, 1972) adds wind information to instability:

```text
SWEAT = 12·Td₈₅₀°C + 20(TT−49) + 2·ff₈₅₀_kt + ff₅₀₀_kt + 125(S + 0.2)
```

where Td₈₅₀°C is 850 hPa dewpoint in °C (set to 0 if negative), TT is Total Totals (second term = 0 if TT < 49), ff₈₅₀ and ff₅₀₀ are wind speeds in knots, and S = sin(dd₅₀₀ − dd₈₅₀) is the directional wind shear. The shear term is activated only when: 130° ≤ dd₈₅₀ ≤ 250°, 210° ≤ dd₅₀₀ ≤ 310°, dd₅₀₀ > dd₈₅₀, ff₈₅₀ ≥ 15 kt, ff₅₀₀ ≥ 15 kt. Thresholds: < 250 slight; 250–300 moderate severe; 300–400 severe thunderstorm; > 400 tornado possible.

### A.4 Helicity and Composite Indices

#### A.4.1 Storm-Relative Helicity (SRH)
SRH (Davies-Jones et al., 1990) measures the streamwise vorticity available for updraft rotation in supercell thunderstorms:

```text
SRH = −∫₀^H (u_sr · dv/dz − v_sr · du/dz) dz     [m² s⁻²]
      u_sr = u − u_storm,    v_sr = v − v_storm
```

Storm motion is estimated via the Bunkers et al. (2000) method: deviate 7.5 m/s to the right of the 0–6 km mean shear vector. The integral is computed over the 0–3 km layer (SRH₀₋₃) for supercell diagnosis and the 0–1 km layer (SRH₀₋₁) for tornado risk. Discrete form: SRH = −Σᵢ[(u_sr,i + u_sr,i+1)(vᵢ₊₁ − vᵢ) − (v_sr,i + v_sr,i+1)(uᵢ₊₁ − uᵢ)]. Thresholds: 0–150 weak; 150–300 moderate; 300–450 significant; > 450 extreme.

#### A.4.2 Supercell Composite Parameter (SCP)
SCP (Thompson et al., 2003) combines three supercell ingredients, each normalised by its threshold value:

```text
SCP = (CAPE/1000) × (SRH/50) × (BWD/20)
BWD = |V(6km) − V(0m)|          (bulk wind difference, m s⁻¹)
```

Conditional zeroing: SCP = 0 if CAPE < 100 J/kg, SRH < 50 m² s⁻², or BWD < 10 m/s. Thresholds: < 0 no threat; 0–1 marginal; 1–4 supercell likely; > 4 intense supercell.

#### A.4.3 Energy-Helicity Index (EHI)
EHI (Hart and Korotky, 1991) combines CAPE and SRH into a tornado proximity index:

```text
EHI = CAPE × SRH / 160 000
```

EHI = 0 for negative SRH (anticyclonic environments do not support right-moving supercells). Thresholds: < 1 no significant threat; 1–2 supercell possible; 2–4 supercell likely; > 4 significant tornado potential.

#### A.4.4 Precipitable Water (PW)
PW is the vertically integrated water vapour column:

```text
PW = (1/g) ∫ q dp  =  ∫ ρ_v dz         [kg m⁻² = mm]
```

where q = w/(1+w) is specific humidity and w = 0.622·e/(p−e) is mixing ratio, with e = 611.2·exp(17.67·Tc/(Tc+243.5)) Pa (Tetens formula, Tc = T−273.15 °C). Two integration paths (height-based and pressure-based) are averaged for accuracy. Thresholds: < 10 very dry; 10–25 dry; 25–45 moist; 45–60 very moist; > 60 extreme.

### A.5 Eddy Dissipation Rate (EDR) Estimation
EDR = ε^(1/3) [m^(2/3) s^(−1)] is the ICAO standard turbulence metric, where ε is the kinetic energy dissipation rate. Three independent methods are implemented; their median is used as the final EDR estimate. Prior to all methods, wind components are high-pass filtered at 0.5 Hz to remove rotor-induced downwash bias.

#### A.5.1 Structure Function Method (SF)
In Kolmogorov's (1941) inertial subrange, the second-order longitudinal structure function follows a 2/3 power law:

```text
D_L(r) = ⟨[u_L(x+r) − u_L(x)]²⟩ = C² · ε^(2/3) · r^(2/3)
```

where r = lag × U/fs is the physical separation (Taylor frozen turbulence hypothesis), U is mean airspeed, fs is sampling rate, and C² = 2.0 is the Kolmogorov constant. A linear regression in log-log space:

```text
log D_L(r) = (2/3) log(ε) + log(C²) + (2/3) log(r)
```

over lags 3–40 samples yields the intercept from which ε^(2/3) = exp(intercept)/C² is extracted. The fitted slope should be 0.667; deviations > 0.4 trigger a confidence warning.

#### A.5.2 Power Spectral Density Method (PSD)
In the inertial subrange the one-dimensional velocity PSD follows:

```text
S(f) = A · (U/2π)^(2/3) · ε^(2/3) · f^(−5/3)
```

where A = 0.55 is the one-dimensional Kolmogorov constant. A Welch PSD (Hanning windows, 50% overlap) is computed, then the slope and intercept are fitted by linear regression in log-log space over f ∈[0.5, fs/4] Hz:

```text
ε^(2/3) = exp(intercept) / (A · (U/2π)^(2/3))
EDR = max(ε^(2/3), 0)^(3/4)
```

The fitted slope should be −5/3 ≈ −1.667; deviations > 0.5 trigger a confidence warning.

#### A.5.3 Variance Method
A fast approximation from vertical wind variance (Lenschow et al., 1994):

```text
ε ≈ (σ_w / C_w)³ / L_outer
EDR = max(ε, 0)^(1/3)
```

where σ_w is the standard deviation of filtered w, C_w = 1.9 is an empirical constant, and L_outer = 100 m is an assumed outer length scale. This method does not require Taylor's hypothesis and is preferred when airspeed < 1 m/s or the window contains fewer than 50 samples.

#### A.5.4 Confidence Assessment
Confidence is assessed from three criteria: (1) inter-method spread σ_methods / μ_methods < 0.30 for high, < 0.60 for medium, otherwise low; (2) spectral slope deviation < 0.2 (SF) and < 0.3 (PSD) from theoretical values; (3) minimum 100 samples. EDR severity thresholds (ICAO / Sharman et al. 2014):

*Table A.2. EDR severity thresholds following ICAO Circular 332 and Sharman et al. (2014).*

| Category | EDR range (m^(2/3) s^(−1)) | Aviation significance |
|---|---|---|
| Calm | 0 – 0.05 | No hazard |
| Light | 0.05 – 0.10 | Slight, non-hazardous |
| Light-to-moderate | 0.10 – 0.22 | Noticeable, objects may move |
| Moderate | 0.22 – 0.34 | Difficult to walk; unsecured items move |
| Moderate-to-severe | 0.34 – 0.45 | Aircraft may briefly lose control |
| Severe | > 0.45 | Large abrupt changes; structural risk |

---