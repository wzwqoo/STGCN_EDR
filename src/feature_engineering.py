import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.80665  # gravitational acceleration  [m s^-2]
R_D = 287.05  # dry air gas constant         [J kg^-1 K^-1]
C_P = 1004.0  # specific heat at const pres  [J kg^-1 K^-1]
KAPPA = R_D / C_P  # Poisson exponent  (~0.2854)
P_REF = 100_000.0  # reference pressure [Pa] (1000 hPa)
R_V     = 461.5      # [J kg^-1 K^-1]
L_V     = 2.501e6    # [J kg^-1]  latent heat of vaporisation
GAMMA_D = G / C_P    # dry adiabatic lapse rate ≈ 0.00976 K/m
MS_TO_KT = 1.94384   # m/s → knots
M_TO_KFT   = 3.28084e-3 # m    → kilofeet  (1000 ft)

ELLROD_THRESHOLDS = {
    "smooth": (0, 4),
    "light": (4, 8),
    "moderate": (8, 12),
    "severe": (12, 999),
}

RI_THRESHOLDS = {
    "turbulent": (-np.inf, 0.0),
    "KH_unstable": (0.0, 0.25),
    "transitional": (0.25, 1.0),
    "stable": (1.0, np.inf),
}

N2_THRESHOLDS = {
    "stable (oscillation)" : (0.0, np.inf),
    "convective instability": (-np.inf, 0.0)
}

CP_THRESHOLDS = {
    "smooth":   (0.0,     1.5e-4),
    "light":    (1.5e-4,  6.0e-4),
    "moderate": (6.0e-4,  1.5e-3),
    "severe":   (1.5e-3,  np.inf),
}

BI_THRESHOLDS = {
    "smooth":   (0.0,     2.0e-4),
    "light":    (2.0e-4,  8.0e-4),
    "moderate": (8.0e-4,  2.0e-3),
    "severe":   (2.0e-3,  np.inf),
}

CAPE_THRESHOLDS = {
    "stable": (-np.inf, 300.0),
    "marginal convection": (300.0, 1000),
    "moderate convection": (1000, 2500),
    "extreme convection": (2500, np.inf),
}

CIN_THRESHOLDS = {
    "no cap":      (-25.0,      0.0),       # easy to initiate
    "weak cap":    (-50.0,    -25.0),       # needs surface heating
    "moderate cap":(-200.0,   -50.0),       # needs strong forcing
    "strong cap":  (-np.inf, -200.0),       # convection very unlikely
}

LI_THRESHOLDS = {
    "stable":              ( 2.0,   np.inf),
    "slightly unstable":   ( 0.0,    2.0),
    "moderately unstable": (-3.0,    0.0),
    "very unstable":       (-6.0,   -3.0),
    "extremely unstable":  (-np.inf,-6.0),
}

SI_THRESHOLDS = {
    "stable":              ( 3.0,   np.inf),
    "slightly unstable":   ( 1.0,    3.0),
    "moderately unstable": (-3.0,    1.0),
    "very unstable":       (-6.0,   -3.0),
    "extremely unstable":  (-np.inf,-6.0),
}

K_THRESHOLDS = {
    "no convection": (-np.inf, 15.0),
    "isolated thunderstorms": (15.0, 25.0),
    "scattered thunderstorms": (25.0, 35.0),
    "numerous thunderstorms": (35.0, 40.0),
    "extreme thunderstorms": (40.0, np.inf),
}

TT_THRESHOLDS = {
    "no convection": (-np.inf, 44.0),
    "isolated thunderstorms": (44.0, 50.0),
    "scattered thunderstorms": (50.0, 55.0),
    "severe thunderstorms": (55.0, 60.0),
    "extreme thunderstorms": (60.0, np.inf),
}

SWEAT_THRESHOLDS = {
    "slight severe": (-np.inf, 250.0),
    "moderate severe": (250.0, 300.0),
    "severe thunderstorm": (300.0, 400.0),
    "tornado possible": (400.0, np.inf),
}

DUTTON_THRESHOLDS = {
    "smooth": (0.0, 20.0),
    "light": (20.0, 30.0),
    "moderate": (30.0, 45.0),
    "severe": (45.0, np.inf),
}

TKE_THRESHOLDS = {
    "calm": (0.0, 0.1),
    "light": (0.1, 0.5),
    "moderate": (0.5, 2.0),
    "severe": (2.0, np.inf),
}

SCORER_THRESHOLDS = {
    "evanescent (no waves)": (-np.inf, 0.0),
    "weak wave activity": (0.0, 1.0e-6),
    "moderate wave activity": (1.0e-6, 1.0e-5),
    "strong wave / trapping": (1.0e-5, np.inf),
}

PW_THRESHOLDS = {
    "very dry": (0.0, 10.0),
    "dry": (10.0, 25.0),
    "moist": (25.0, 45.0),
    "very moist": (45.0, 60.0),
    "extreme": (60.0, np.inf),
}

SRH_THRESHOLDS = {
    "weak": (0.0, 150.0),
    "moderate": (150.0, 300.0),
    "significant": (300.0, 450.0),
    "extreme": (450.0, np.inf),
}

SRH_NEGATIVE_THRESHOLDS = {
    "anticyclonic weak": (-150.0, 0.0),
    "anticyclonic significant": (-np.inf, -150.0),
}

EHI_THRESHOLDS = {
    "no significant threat": (-np.inf, 1.0),
    "supercell possible": (1.0, 2.0),
    "supercell likely": (2.0, 4.0),
    "significant tornado": (4.0, np.inf),
}

SCP_THRESHOLDS = {
    "no threat": (-np.inf, 0.0),
    "marginal supercell": (0.0, 1.0),
    "supercell likely": (1.0, 4.0),
    "intense supercell": (4.0, np.inf),
}

BWD_THRESHOLDS = {
    "weak shear": (0.0, 10.0),  # m/s
    "moderate shear": (10.0, 20.0),
    "strong shear": (20.0, np.inf),
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def potential_temperature(T_K: np.ndarray, p_Pa: np.ndarray) -> np.ndarray:
    """
    Potential temperature  θ = T × (P0/p)^κ

    Parameters
    ----------
    T_K  : absolute temperature [K]
    p_Pa : pressure [Pa]

    Returns
    -------
    θ [K]
    """
    return T_K * (P_REF / p_Pa) ** KAPPA


def saturation_vapour_pressure(T_K):
    """
    Tetens formula.  T in Kelvin, returns e_s in Pa.
    """
    T_C = T_K - 273.15
    return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))


def mixing_ratio_from_dewpoint(Td_K, p_Pa):
    """
    Saturation mixing ratio at dewpoint temperature [kg/kg].
    """
    e = saturation_vapour_pressure(Td_K)
    return 0.622 * e / (p_Pa - e)


def lcl_temperature(T_K, Td_K):
    """
    Bolton (1980) approximation for LCL temperature [K].
    """
    return 1.0 / (1.0 / (Td_K - 56.0) + np.log(T_K / Td_K) / 800.0) + 56.0


def moist_adiabatic_lapse_rate(T_K, p_Pa):
    """
    Saturated (pseudo) adiabatic lapse rate [K/m].

        Γs = g · (1 + Lv·ws / Rd·T)
               / (Cp + Lv²·ws / Rv·T²)

    where ws = saturation mixing ratio.
    """
    ws = mixing_ratio_from_dewpoint(T_K, p_Pa)  # sat at T
    num = G * (1.0 + L_V * ws / (R_D * T_K))
    den = C_P + L_V ** 2 * ws / (R_V * T_K ** 2)
    return num / den


def lift_parcel(T_sfc, Td_sfc, p_sfc, z_sfc, z_levels, p_levels):
    """
    Lift a surface parcel to each level in z_levels.

    Returns
    -------
    T_parcel : np.ndarray  parcel temperature at each level [K]
    z_lcl    : float       LCL altitude [m]
    """
    # ── LCL ──
    T_lcl = lcl_temperature(T_sfc, Td_sfc)
    # height of LCL via dry adiabat: T drops at Γd from surface
    z_lcl = z_sfc + (T_sfc - T_lcl) / GAMMA_D

    T_parcel = np.empty(len(z_levels))

    for i, (z, p) in enumerate(zip(z_levels, p_levels)):
        if z <= z_lcl:
            # dry adiabatic below LCL
            T_parcel[i] = T_sfc - GAMMA_D * (z - z_sfc)
        else:
            # moist adiabatic above LCL — integrate upward in small steps
            if i == 0 or z_levels[i - 1] < z_lcl:
                # first level above LCL: start from LCL state
                T_p = T_lcl
                z_p = z_lcl
                p_p = p_sfc * (T_lcl / T_sfc) ** (G / (R_D * GAMMA_D))
            else:
                T_p = T_parcel[i - 1]
                z_p = z_levels[i - 1]
                p_p = p_levels[i - 1]

            # integrate with small dz steps (50 m)
            dz_step = 50.0
            n_steps = max(1, int((z - z_p) / dz_step))
            dz = (z - z_p) / n_steps
            for _ in range(n_steps):
                gamma_s = moist_adiabatic_lapse_rate(T_p, p_p)
                T_p -= gamma_s * dz
                p_p -= p_p * G / (R_D * T_p) * dz  # hydrostatic
            T_parcel[i] = T_p

    return T_parcel, z_lcl


def estimate_storm_motion(
        u: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
        method: str = "bunkers",
):
    """
    Estimate supercell storm motion vector.

    Two methods:
      'mean'    : simple pressure-weighted mean wind 0–6 km
                  (quick, less accurate)
      'bunkers' : Bunkers et al. (2000) — mean wind 0–6 km
                  deviated 7.5 m/s to the right of the shear vector
                  (standard operational method for right-moving supercells)

    Parameters
    ----------
    u, v : wind components at each level [m/s]
    z    : altitude at each level [m]
    method : 'bunkers' or 'mean'

    Returns
    -------
    (u_storm, v_storm) in m/s
    """
    # mean wind in the 0–6 km layer
    mask_6km = z <= 6000.0
    if mask_6km.sum() < 2:
        # fall back to full-column mean if sounding is shallow
        mask_6km = np.ones(len(z), dtype=bool)

    u_mean = float(np.mean(u[mask_6km]))
    v_mean = float(np.mean(v[mask_6km]))

    if method == "mean":
        return u_mean, v_mean

    # Bunkers: shear vector from 0–0.5 km mean to 5.5–6 km mean
    mask_low = z <= 500.0
    mask_high = (z >= 5500.0) & (z <= 6500.0)

    if mask_low.sum() < 1 or mask_high.sum() < 1:
        # sounding too shallow for Bunkers — fall back to mean wind
        return u_mean, v_mean

    u_low = float(np.mean(u[mask_low]))
    v_low = float(np.mean(v[mask_low]))
    u_high = float(np.mean(u[mask_high]))
    v_high = float(np.mean(v[mask_high]))

    # shear vector (low → high)
    du_shear = u_high - u_low
    dv_shear = v_high - v_low
    shear_mag = np.sqrt(du_shear ** 2 + dv_shear ** 2)

    if shear_mag < 0.1:
        return u_mean, v_mean

    # right-mover: deviate 7.5 m/s 90° to the right of the shear vector
    # rotating shear vector 90° clockwise: (dx, dy) → (dy, -dx)
    D = 7.5  # m/s deviation magnitude
    u_storm = u_mean + D * (dv_shear / shear_mag)
    v_storm = v_mean + D * (-du_shear / shear_mag)

    return float(u_storm), float(v_storm)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def ellrod_index(
        # self
        x0, y0, z0, u0, v0,
        # neighbour 1  (same altitude, any direction)
        x1, y1, u1, v1,
        # neighbour 2  (same altitude, non-collinear with N1)
        x2, y2, u2, v2,
        # UAV below  (for vertical wind shear)
        z_low, u_low, v_low,
):
    """
    1. Ellrod Turbulence Index (TI1) on a 2-D horizontal grid.
    ssuming 2 neighbours on same altitude and 1 neighbour in same column

        EI = VWS × DEF           where DEF = sqrt(DST² + DSH²)
        DST = ∂u/∂x − ∂v/∂y     (stretching deformation)
        DSH = ∂v/∂x + ∂u/∂y     (shearing deformation)
        VWS = sqrt(Δu² + Δv²) / Δz   (vertical wind shear magnitude)
        Δu = (∂u/∂x)·Δx + (∂u/∂y)·Δy
        Δv = (∂v/∂x)·Δx + (∂v/∂y)·Δy

        → Δu/Δx is not ∂u/∂x — it is ∂u/∂x·(Δx/Δx) + ∂u/∂y·(Δy/Δx)
        → Δv, Δu, Δx, Δy are mesaured value and gradients are unknowns, 4 equation(2 neighbours needed) for 4 unknowns

    Returns
    -------
    EI in 10^-7 s^-2,  shape (ny, nx)
    """

    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0
    du1, dv1 = u1 - u0, v1 - v0
    du2, dv2 = u2 - u0, v2 - v0

    #   M · [∂u/∂x, ∂u/∂y]ᵀ = [Δu1, Δu2]ᵀ
    #   M · [∂v/∂x, ∂v/∂y]ᵀ = [Δv1, Δv2]ᵀ
    M = np.array([[dx1, dy1],[dx2, dy2]], dtype=float)
    M_inv = np.linalg.inv(M)

    du_dx, du_dy = M_inv @ [du1, du2]
    dv_dx, dv_dy = M_inv @ [dv1, dv2]

    DST = du_dx - dv_dy          # stretching deformation  [s^-1]
    DSH = dv_dx + du_dy          # shearing deformation    [s^-1]
    DEF = np.sqrt(DST**2 + DSH**2)
    VWS = np.sqrt((u0 - u_low)**2 + (v0 - v_low)**2) / (z0 - z_low)   # [s^-1]
    EI_raw = VWS * DEF                               # [s^-2]
    EI_scaled = EI_raw * 1e7                           # conventional × 10^-7 unit

    return EI_scaled

def richardson_number(
        # self (upper level)
        z0, u0, v0, T0, p0,
        # UAV directly below (lower level)
        z_low, u_low, v_low, T_low, p_low,
):
    """
    1. Richardson Number between two altitude levels.
    assuming one neighbour in same column

        Ri = (g / θ_mean) × (Δθ / Δz) / [(Δu/Δz)² + (Δv/Δz)²]

    Temperature + pressure at 2+ heights → compute ∂θ/∂z → Richardson numerator (stability)
    Wind speed/direction at 2+ heights → compute ∂u/∂z → Richardson denominator (shear)
    2. Brown index VWS² / (Ri + 1), Combines Ri with absolute VWS
    3. Colson-Panofsky (CP), VWS² × (1 − Ri), Emphasises Ri near 0, Active when shear high, atmosphere near-neutral
    4. Brunt-Väisälä = (g / θ_mean) × (Δθ / Δz)

    Returns
    -------
    Ri  dimensionless  (clipped to ±100 to avoid division-noise)
    """
    theta_upper = potential_temperature(T0, p0)
    theta_lower = potential_temperature(T_low, p_low)
    theta_mean = 0.5 * (theta_lower + theta_upper)
    delta_theta = theta_upper - theta_lower

    # buoyancy term (numerator), Brunt-Väisälä
    dz = z0 - z_low
    N2 = (G / theta_mean) * (delta_theta / dz)

    # wind shear squared (denominator)
    du_dz = (u0 - u_low) / dz
    dv_dz = (v0 - v_low) / dz
    S2 = du_dz ** 2 + dv_dz ** 2

    # Guard against zero shear — replace with tiny value to signal infinite Ri
    epsilon = 1e-12
    S2_safe = np.where(S2 < epsilon, epsilon, S2)

    ri = N2 / S2_safe

    # Brown index and Colson-Panofsky (CP)
    VWS = np.sqrt((u0 - u_low) ** 2 + (v0 - v_low) ** 2) / (z0 - z_low)  # [s^-1]
    bi = VWS**2 / (ri + 1)
    cp = VWS**2 * (1 - ri)
    return {
            "richardson_number": np.clip(ri, -100.0, 100.0),
            "Brunt-Väisälä ": N2,
            "Brown_index": bi,
            "Colson-Panofsky": cp
        }

def cape_cin(
        z: np.ndarray,  # altitude AGL at each UAV level [m],  shape (n,)
        T_K: np.ndarray,  # air temperature [K],                  shape (n,)
        p_Pa: np.ndarray,  # pressure [Pa],                        shape (n,)
        Td_K: np.ndarray,  # dewpoint temperature [K],             shape (n,)
        sfc_idx: int = 0,  # index of the surface (parcel source) level
):
    """
    1. CAPE= ∫(Tv_parcel − Tv_env) dz
    needs only verticle neighbours, typically 5-8 to be usable
    2. and CIN = −∫(Tv_env − Tv_parcel) dz from a UAV column sounding.
    3. LI = T_env(500hPa) - T_parcel(500hPa), negative = parcel warmer than environment = unstable
    For a UAV swarm, this means LI is only computable if you have fixed-wing UAVs or tethered sondes reaching above ~5500 m, or if you blend the UAV column with radiosonde data above the UAV ceiling.
    4. SI lifts from 850 hPa (~1500 m)

    Parameters
    ----------
    z     : altitudes from surface UAV up to highest UAV [m]
    T_K   : environment temperature at each level [K]
    p_Pa  : environment pressure at each level [Pa]
    Td_K  : dewpoint at each level [K]
    sfc_idx : which level to lift the parcel from (default 0 = surface)

    """
    # ensure sorted surface → top
    order = np.argsort(z)
    z, T_K, p_Pa, Td_K = z[order], T_K[order], p_Pa[order], Td_K[order]

    n = len(z)
    if n < 2:
        raise ValueError("Need at least 2 altitude levels to compute CAPE.")

    # ── lift parcel from surface level ──
    T_parcel, z_lcl = lift_parcel(
        T_sfc=T_K[sfc_idx],
        Td_sfc=Td_K[sfc_idx],
        p_sfc=p_Pa[sfc_idx],
        z_sfc=z[sfc_idx],
        z_levels=z,
        p_levels=p_Pa,
    )

    # ── buoyancy at each level ──
    # b = g · (T_parcel − T_env) / T_env
    buoyancy = G * (T_parcel - T_K) / T_K  # [m/s²]

    # ── find LFC (first level where parcel becomes positively buoyant) ──
    z_lfc = None
    for i in range(1, n):
        if buoyancy[i - 1] <= 0 and buoyancy[i] > 0:
            # linear interpolation to exact crossing
            frac = buoyancy[i - 1] / (buoyancy[i - 1] - buoyancy[i])
            z_lfc = z[i - 1] + frac * (z[i] - z[i - 1])
            break

    # ── find EL (last level where parcel is still positively buoyant) ──
    z_el = None
    for i in range(n - 1, 0, -1):
        if buoyancy[i] > 0 and buoyancy[i - 1] <= 0:
            frac = buoyancy[i] / (buoyancy[i] - buoyancy[i - 1])
            z_el = z[i] - frac * (z[i] - z[i - 1])
            break

    # ── integrate CAPE and CIN using trapezoid rule ──
    CAPE = 0.0
    CIN = 0.0

    for i in range(1, n):
        dz_layer = z[i] - z[i - 1]
        b_mean = 0.5 * (buoyancy[i] + buoyancy[i - 1])

        if b_mean > 0:
            # positive buoyancy → only count above LFC
            if z_lfc is not None and z_lfc <= z[i - 1] <= z_el:
                CAPE += b_mean * dz_layer
        else:
            # negative buoyancy → only count below LFC for CIN
            if z_lfc is None or z[i] <= (z_lfc if z_lfc else np.inf):
                CIN += b_mean * dz_layer  # already negative

    # ── Lifted Index: interpolate T_env and T_parcel to 500 hPa ──
    LI = None
    p_500 = 50_000.0  # 500 hPa in Pa
    if p_Pa.min() <= p_500 <= p_Pa.max():
        # pressure decreases with height so flip for np.interp
        T_env_500 = float(np.interp(p_500, p_Pa[::-1], T_K[::-1]))
        T_parcel_500 = float(np.interp(p_500, p_Pa[::-1], T_parcel[::-1]))
        LI = T_env_500 - T_parcel_500
    else:
        if p_Pa.min() > p_500:
            print("unavailable — sounding does not reach 500 hPa")
            LI = None

    SI = None
    p_850 = 85_000.0  # 500 hPa in Pa
    if p_Pa.min() <= p_850 <= p_Pa.max():
        # pressure decreases with height so flip for np.interp
        T_env_850 = float(np.interp(p_850, p_Pa[::-1], T_K[::-1]))
        T_parcel_850 = float(np.interp(p_850, p_Pa[::-1], T_parcel[::-1]))
        z_850 = float(np.interp(p_850, p_Pa[::-1], z[::-1]))
        p_850_val = p_850
        # lift from 850 hPa upward through the remaining levels
        mask_above = z >= z_850
        if mask_above.sum() >= 2:
            z_above = z[mask_above]
            p_above = p_Pa[mask_above]
            T_parcel_si, _ = lift_parcel(
                T_sfc=T_env_850,
                Td_sfc=T_parcel_850,
                p_sfc=p_850_val,
                z_sfc=z_850,
                z_levels=z_above,
                p_levels=p_above,
            )
            T_parcel_si_500 = float(
                np.interp(p_500, p_above[::-1], T_parcel_si[::-1])
            )
            SI = T_env_500 - T_parcel_si_500
    else:
        if p_Pa.min() > p_850:
            print("unavailable — sounding does not reach 850 hPa")
            LI = None

    return {
        "CAPE_J_kg": CAPE,
        "CIN_J_kg": CIN,
        "LI": LI,
        "SI":SI
    }


def sweat_index(
    p_Pa:  np.ndarray,
    T_K:   np.ndarray,
    Td_K:  np.ndarray,
    u_ms:  np.ndarray,   # zonal wind [m/s]
    v_ms:  np.ndarray,   # meridional wind [m/s]
):
    """
    1. Total Totals Index from a UAV column. Vertical levels needed        : 850, 500 hPa  (2 UAVs minimum)
    2. K-Index from a UAV column.     K = (T850 − T500) + Td850 − (T700 − Td700)
                                        = lapse rate term + low-level moisture − mid-level dryness
    Vertical levels needed        : 850, 700, 500 hPa  (3 UAVs minimum)
    3. Severe Weather Threat (SWEAT) Index from a UAV column.   SWEAT = 12·Td850°C  +  20·(TT−49)  +  2·ff850kt  +  ff500kt  +  125·(S+0.2)
    Vertical levels needed        : 850, 500 hPa  (2 UAVs minimum)
                                    u and v wind required at both level

    Parameters
    ----------
    p_Pa  : pressure [Pa], sorted surface → top
    T_K   : temperature [K]
    Td_K  : dewpoint [K]
    u_ms  : zonal wind [m/s]
    v_ms  : meridional wind [m/s]
    """

    p_Pa = np.asarray(p_Pa, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    Td_K = np.asarray(Td_K, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)

    # ── interpolate to 850 and 500 hPa ──
    T850 = np.interp(85_000, p_Pa[::-1], T_K[::-1])
    T700 = np.interp(70_000, p_Pa[::-1], T_K[::-1])
    T500 = np.interp(50_000, p_Pa[::-1], T_K[::-1])
    Td850 = np.interp(85_000, p_Pa[::-1], Td_K[::-1])
    Td700 = np.interp(70_000, p_Pa[::-1], Td_K[::-1])
    Td500 = np.interp(50_000, p_Pa[::-1], Td_K[::-1])
    u850 = np.interp(85_000, p_Pa[::-1], u_ms[::-1])
    v850 = np.interp(85_000, p_Pa[::-1], v_ms[::-1])
    u500 = np.interp(50_000, p_Pa[::-1], u_ms[::-1])
    v500 = np.interp(50_000, p_Pa[::-1], v_ms[::-1])

    # ── wind speed [knots] and direction [°] ──
    ff850_kt = np.sqrt(u850 ** 2 + v850 ** 2) * MS_TO_KT
    ff500_kt = np.sqrt(u500 ** 2 + v500 ** 2) * MS_TO_KT
    dd850 = np.degrees(np.arctan2(-u850, -v850)) % 360
    dd500 = np.degrees(np.arctan2(-u500, -v500)) % 360

    # ── Total Totals (needed inside SWEAT) ──
    VT = T850 - T500  # vertical lapse rate term
    CT = Td850 - T500  # cross moisture term
    TT = VT + CT

    # ── Term 1: low-level moisture ──
    Td850_C = Td850 - 273.15
    term1 = 12.0 * Td850_C

    # ── Term 2: instability ──
    term2 = 20.0 * (TT - 49.0)

    # ── Terms 3 & 4: wind speed ──
    term3 = 2.0 * ff850_kt
    term4 = ff500_kt

    # ── Term 5: directional wind shear (veering) ──
    shear_active = (
            130 <= dd850 <= 250 and
            210 <= dd500 <= 310 and
            (dd500 - dd850) > 0 and
            ff850_kt >= 15 and
            ff500_kt >= 15
    )
    if shear_active:
        S = np.sin(np.radians(dd500 - dd850))
        term5 = 125.0 * (S + 0.2)
    else:
        S = None
        term5 = 0.0

    SWEAT = term1 + term2 + term3 + term4 + term5
    K = (T850 - T500) + (Td850 - 273.15) - (T700 - Td700)

    return {
        "SWEAT": SWEAT,
        "TT": TT,
        "K": K,
    }


def dutton_index(
    # vertical pair — speed shear
    z0, u0, v0, T0,  # self  (upper level)
    z_low, u_low, v_low,  # 1 UAV below (same column)
    # horizontal pair — temperature gradient along wind
    T_n1, x_n1, y_n1,  # neighbour 1, same altitude as self
    T_n2, x_n2, y_n2,  # neighbour 2, same altitude as self
    x0=0.0, y0=0.0,  # self horizontal position (default origin)
):
    """
    Dutton Turbulence Index from Dutton (1969).

        Ei = 1.5 · S  +  0.5 · ΔT

    where:
        S   = vertical speed shear  [kt / 1000 ft]
        ΔT  = horizontal temperature change along the wind direction [K or °C]
              estimated by projecting the horizontal T gradient onto the
              wind vector at the upper level

    Horizontal neighbours needed : 2 (same altitude, non-collinear,
                                      for horizontal T gradient)
    Vertical neighbours needed   : 1 below (for speed shear)

    Parameters
    ----------
    z0, u0, v0, T0       : self position, wind, temperature
    z_low, u_low, v_low  : lower UAV wind (T not needed for Dutton)
    T_n1, x_n1, y_n1     : temperature and position of horizontal neighbour 1
    T_n2, x_n2, y_n2     : temperature and position of horizontal neighbour 2
    x0, y0               : self horizontal position [m]
    """
    # ── speed shear S [kt / 1000 ft] ──
    spd0 = np.sqrt(u0 ** 2 + v0 ** 2)
    spd_low = np.sqrt(u_low ** 2 + v_low ** 2)
    dspd_ms = spd0 - spd_low  # m/s
    dz_kft = (z0 - z_low) * M_TO_KFT  # convert m → 1000 ft
    S = (dspd_ms * MS_TO_KT) / dz_kft  # kt / 1000 ft

    # ── horizontal temperature gradient (2×2 system) ──
    dx1, dy1 = x_n1 - x0, y_n1 - y0
    dx2, dy2 = x_n2 - x0, y_n2 - y0
    dT1, dT2 = T_n1 - T0, T_n2 - T0

    det = dx1 * dy2 - dx2 * dy1
    if abs(det) < 1e-6:
        raise ValueError("Horizontal neighbours are collinear — cannot solve for ∇T.")

    dT_dx = (dy2 * dT1 - dy1 * dT2) / det
    dT_dy = (-dx2 * dT1 + dx1 * dT2) / det

    # ΔT along wind direction (unit wind vector · ∇T × reference distance)
    # Dutton used ~3° latitude ≈ 333 km; we normalise to per-metre gradient
    # and report ΔT over the actual horizontal spacing for interpretability
    wind_mag = np.sqrt(u0 ** 2 + v0 ** 2)
    if wind_mag < 0.1:
        dT_along = 0.0  # calm — no preferred direction
    else:
        ux, uy = u0 / wind_mag, v0 / wind_mag  # unit wind vector
        # gradient in K/m along wind; multiply by reference distance (333 km)
        dT_along = (dT_dx * ux + dT_dy * uy) * 333_000.0

    Ei = 1.5 * abs(S) + 0.5 * abs(dT_along)

    return {
        "Ei": float(Ei),
    }


def turbulent_kinetic_energy(
        u_samples: np.ndarray,  # time-series of u-wind [m/s], shape (n,)
        v_samples: np.ndarray,  # time-series of v-wind [m/s], shape (n,)
        w_samples: np.ndarray,  # time-series of w-wind [m/s], shape (n,)
):
    """
    Turbulent Kinetic Energy per unit mass from UAV time-series samples.

        TKE = ½ · (σu² + σv² + σw²)

    where σ² is the variance of each wind component about its mean.
    This is the standard Reynolds-decomposition definition:
        u = ū + u'   →   σu² = mean(u'²)

    No spatial neighbours needed — computed entirely from the UAV's own
    anemometer time series sampled at ≥5 Hz over a ≥30 s window.

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : 0

    Practical note
    --------------
    Multi-rotor UAVs introduce rotor-wash contamination into w_samples.
    Apply a high-pass filter (cutoff ~0.1 Hz) before calling this function
    to remove low-frequency platform motion, and correct for rotor-induced
    w bias (~0.3–0.5 m/s downwash) using manufacturer calibration data.

    Parameters
    ----------
    u_samples : east-west wind time series [m/s]
    v_samples : north-south wind time series [m/s]
    w_samples : vertical wind time series [m/s]
    """
    u = np.asarray(u_samples, dtype=float)
    v = np.asarray(v_samples, dtype=float)
    w = np.asarray(w_samples, dtype=float)

    if len(u) < 10:
        raise ValueError("Need at least 10 samples for a meaningful TKE estimate.")

    var_u = float(np.var(u, ddof=1))
    var_v = float(np.var(v, ddof=1))
    var_w = float(np.var(w, ddof=1))

    TKE = 0.5 * (var_u + var_v + var_w)

    # turbulence intensity: TKE normalised by mean wind KE
    mean_spd = float(np.sqrt(np.mean(u) ** 2 + np.mean(v) ** 2))
    TI = np.sqrt(2 * TKE / 3) / mean_spd if mean_spd > 0.1 else np.nan

    return {
        "TKE_J_kg": TKE,
        "TI":TI
    }


def scorer_parameter(
        z: np.ndarray,  # altitude at each UAV level [m],    shape (n,) n≥3
        T_K: np.ndarray,  # temperature [K],                   shape (n,)
        p_Pa: np.ndarray,  # pressure [Pa],                     shape (n,)
        u_ms: np.ndarray,  # zonal wind [m/s],                  shape (n,)
        v_ms: np.ndarray,  # meridional wind [m/s],             shape (n,)
):
    """
    Scorer Parameter profile from a UAV column.

        l²(z) = N²(z) / U²(z)  −  (1/U(z)) · d²U/dz²

    where:
        N²   = (g/θ) · dθ/dz          Brunt-Väisälä frequency squared
        U    = wind speed magnitude    √(u² + v²)
        d²U/dz²                        curvature of the wind speed profile

    Physical meaning
    ----------------
    Mountain (gravity) waves propagate vertically only where l² > 0.
    If l² decreases with height, waves become trapped in a duct and
    eventually break, releasing turbulent kinetic energy.
    l² < 0 means evanescent waves — energy cannot propagate upward.

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : 3 minimum (self + 2 below) for d²U/dz²
                                   More levels improve curvature accuracy.

    Parameters
    ----------
    z    : altitudes sorted surface → top [m]
    T_K  : temperature [K]
    p_Pa : pressure [Pa]
    u_ms : zonal wind [m/s]
    v_ms : meridional wind [m/s]
    """
    z = np.asarray(z, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    p_Pa = np.asarray(p_Pa, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)

    n = len(z)
    if n < 3:
        raise ValueError(
            f"Scorer parameter requires ≥3 altitude levels; got {n}. "
            "Add more UAVs to the column."
        )

    # ensure sorted surface → top
    order = np.argsort(z)
    z, T_K, p_Pa, u_ms, v_ms = z[order], T_K[order], p_Pa[order], \
        u_ms[order], v_ms[order]

    # ── potential temperature and N² ──
    theta = potential_temperature(T_K, p_Pa)
    dtheta_dz = np.gradient(theta, z)
    N2 = (G / theta) * dtheta_dz  # [rad² s⁻²]

    # ── wind speed and its second derivative ──
    U = np.sqrt(u_ms ** 2 + v_ms ** 2)  # [m/s]
    d2U_dz2 = np.gradient(np.gradient(U, z), z)  # [s⁻¹]  second deriv

    # ── Scorer parameter ──
    U_safe = np.where(np.abs(U) < 0.5, 0.5, U)  # avoid /0 in calm air
    l2 = N2 / U_safe ** 2 - d2U_dz2 / U_safe  # [m⁻²]

    # ── wave trapping: l² decreasing with height ──
    dl2_dz = np.gradient(l2, z)
    trapping = bool(np.any(dl2_dz < 0) and np.any(l2 > 0))
    trap_level = None
    for i in range(len(l2) - 1):
        if l2[i] > 0 and dl2_dz[i] < -1e-10:
            trap_level = float(z[i])
            break

    return {
        "l2_m-2": l2,  # full profile [m⁻²]
        "N2_rad2_s-2": N2,  # full profile
        "U_ms": U,  # wind speed profile
        "d2U_dz2": d2U_dz2,  # curvature profile
        "z_m": z,
        "wave_trapping": trapping,
        "trap_level_m": trap_level,
        "l2_surface": float(l2[0]),
        "n_levels": n,
    }


def precipitable_water(
        z: np.ndarray,  # altitude at each UAV level [m],    shape (n,)
        p_Pa: np.ndarray,  # pressure [Pa],                     shape (n,)
        T_K: np.ndarray,  # temperature [K],                   shape (n,)
        Td_K: np.ndarray,  # dewpoint temperature [K],          shape (n,)
):
    """
    Precipitable Water column from a UAV vertical sounding.

        PW = ∫ ρ_v dz  =  (1/g) ∫ q dp

    integrated from surface to top of sounding using the trapezoid rule.

    Two equivalent integration paths are used and averaged:
        Height-based : ∫ ρ_v dz   (direct, using UAV altitudes)
        Pressure-based: (1/g) ∫ q dp  (standard meteorological form)

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : 2 minimum; 6+ recommended for accuracy

    Parameters
    ----------
    z    : altitudes sorted surface → top [m]
    p_Pa : pressure [Pa]
    T_K  : temperature [K]
    Td_K : dewpoint [K]
    """
    z = np.asarray(z, dtype=float)
    p_Pa = np.asarray(p_Pa, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    Td_K = np.asarray(Td_K, dtype=float)

    if len(z) < 2:
        raise ValueError("Need at least 2 altitude levels for PW.")

    # ensure sorted surface → top
    order = np.argsort(z)
    z, p_Pa, T_K, Td_K = z[order], p_Pa[order], T_K[order], Td_K[order]

    # ── mixing ratio and vapour density at each level ──
    w = mixing_ratio_from_dewpoint(Td_K, p_Pa)  # kg/kg
    q = w / (1.0 + w)  # specific humidity [kg/kg]
    rho = p_Pa / (R_D * T_K)  # dry air density  [kg/m³]
    rho_v = q * rho  # vapour density   [kg/m³]

    # ── height-based integration (trapezoid) ──
    PW_z = float(np.trapezoid(rho_v, z))  # [kg/m²]

    # ── pressure-based integration (1/g ∫ q dp) ──
    # pressure decreases upward so flip sign
    PW_p = float(np.abs(np.trapezoid(q, p_Pa)) / G)  # [kg/m²]

    # average the two estimates
    PW = 0.5 * (PW_z + PW_p)

    # ── layer contributions ──
    layer_pw = []
    for i in range(len(z) - 1):
        lyr = float(np.trapezoid(rho_v[i:i + 2], z[i:i + 2]))
        layer_pw.append({
            "z_bottom_m": float(z[i]),
            "z_top_m": float(z[i + 1]),
            "pw_mm": lyr,
        })

    return {
        "PW_kg_m2": PW,  # = mm of liquid water
    }


def storm_relative_helicity(
        z: np.ndarray,  # altitude at each UAV level [m],   shape (n,)
        u_ms: np.ndarray,  # zonal wind [m/s],                 shape (n,)
        v_ms: np.ndarray,  # meridional wind [m/s],            shape (n,)
        z_top: float = 3000.0,  # integration depth [m]  (3 km default)
        u_storm: float = None,  # storm motion u [m/s]  (None = auto)
        v_storm: float = None,  # storm motion v [m/s]  (None = auto)
        storm_motion_method: str = "bunkers",
):
    """
        SRH = -∫0_h  (u_sr · dv/dz  −  v_sr · du/dz)  dz

    u_sr and v_Sr is storm relative wind, need to know how fast storm is traveling
    Each UAV reports its local barometric pressure. The swarm's ground station uses a "Gradient Descent" algorithm to find the center of the low-pressure system (the storm's eye/core).
    By tracking how that pressure center moves across the GPS map over 5 minutes, the system calculates C (the Storm Motion).
    todo: storm tracking to be implemented

    Equivalently (and numerically stabler via the discrete form):

        SRH = Σ  (u_sr[i] + u_sr[i+1]) · (v[i+1] − v[i])
                -(v_sr[i] + v_sr[i+1]) · (u[i+1] − u[i])

    This is the signed area swept by the storm-relative hodograph
    between the surface and z_top.

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : ≥4 UAVs spanning 0 → z_top
                                   (more = better hodograph resolution)

    Parameters
    ----------
    z          : altitudes sorted surface → top [m]
    u_ms       : zonal wind [m/s]
    v_ms       : meridional wind [m/s]
    z_top      : upper integration limit [m]  (3000 for SRH0-3, 1000 for SRH0-1)
    u_storm    : storm motion zonal component [m/s]  (None = estimate from data)
    v_storm    : storm motion meridional component [m/s]
    storm_motion_method : 'bunkers' or 'mean'

    Returns
    -------
    dict with SRH [m² s⁻²], storm motion, category
    """
    z = np.asarray(z, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)

    # sort surface → top
    order = np.argsort(z)
    z, u_ms, v_ms = z[order], u_ms[order], v_ms[order]

    if len(z) < 2:
        raise ValueError("Need at least 2 altitude levels for SRH.")

    if z.max() < z_top * 0.8:
        raise ValueError(
            f"Sounding top ({z.max():.0f} m) is well below integration "
            f"depth ({z_top:.0f} m). Add UAVs at higher altitudes."
        )

    # ── storm motion ──
    if u_storm is None or v_storm is None:
        u_storm, v_storm = estimate_storm_motion(
            u_ms, v_ms, z, method=storm_motion_method
        )

    # ── clip to integration layer ──
    mask = z <= z_top
    # interpolate wind to exactly z_top if needed
    if z[mask].max() < z_top and mask.sum() < len(z):
        idx = np.searchsorted(z, z_top)
        # linear interpolation at z_top
        frac = (z_top - z[idx - 1]) / (z[idx] - z[idx - 1])
        u_top = u_ms[idx - 1] + frac * (u_ms[idx] - u_ms[idx - 1])
        v_top = v_ms[idx - 1] + frac * (v_ms[idx] - v_ms[idx - 1])
        z_lyr = np.append(z[mask], z_top)
        u_lyr = np.append(u_ms[mask], u_top)
        v_lyr = np.append(v_ms[mask], v_top)
    else:
        z_lyr = z[mask]
        u_lyr = u_ms[mask]
        v_lyr = v_ms[mask]

    # ── storm-relative wind ──
    u_sr = u_lyr - u_storm
    v_sr = v_lyr - v_storm

    # ── discrete helicity sum (trapezoid on hodograph) ──
    SRH = 0.0
    for i in range(len(z_lyr) - 1):
        # negative sign follows Davies-Jones (1990):
        # positive SRH = cyclonic (veering) profile = right-moving supercell
        SRH -= ((u_sr[i] + u_sr[i + 1]) * (v_lyr[i + 1] - v_lyr[i])
                - (v_sr[i] + v_sr[i + 1]) * (u_lyr[i + 1] - u_lyr[i]))

    SRH = float(SRH)

    return {
        "SRH_m2_s2": SRH,
    }


def bulk_wind_difference(
        z: np.ndarray,
        u_ms: np.ndarray,
        v_ms: np.ndarray,
        z_bottom: float = 0.0,
        z_top: float = 6000.0,
):
    """
    Bulk Wind Difference (BWD) — vector wind difference between two levels.

        BWD = |V(z_top) − V(z_bottom)|   [m/s]

    Used in SCP as a measure of deep-layer vertical wind shear.
    Standard layer: surface (0 m) to 6 km.

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : 2 (one near z_bottom, one near z_top)
    """
    z = np.asarray(z, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)

    order = np.argsort(z)
    z, u_ms, v_ms = z[order], u_ms[order], v_ms[order]

    u_bot = float(np.interp(z_bottom, z, u_ms))
    v_bot = float(np.interp(z_bottom, z, v_ms))
    u_top_v = float(np.interp(z_top, z, u_ms))
    v_top_v = float(np.interp(z_top, z, v_ms))

    BWD = float(np.sqrt((u_top_v - u_bot) ** 2 + (v_top_v - v_bot) ** 2))

    return {
        "BWD_ms": BWD,
    }


def energy_helicity_index(
        CAPE: float,  # J/kg  (from cape_cin())
        z: np.ndarray,  # altitude profile [m]
        u_ms: np.ndarray,  # zonal wind profile [m/s]
        v_ms: np.ndarray,  # meridional wind profile [m/s]
        z_srh_top: float = 3000.0,
        u_storm: float = None,
        v_storm: float = None,
        storm_motion_method: str = "bunkers",
):
    """
    Energy-Helicity Index.

        EHI = CAPE × SRH / 160 000

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : same as SRH (≥4 UAVs to z_srh_top)
                                   + enough for CAPE (≥5 UAVs to ~10 km)

    Parameters
    ----------
    CAPE      : J/kg — pass result from cape_cin()["CAPE_J_kg"]
    z, u_ms, v_ms : wind profile for SRH
    z_srh_top : SRH integration depth [m]  default 3000
    """
    srh_result = storm_relative_helicity(
        z, u_ms, v_ms,
        z_top=z_srh_top,
        u_storm=u_storm,
        v_storm=v_storm,
        storm_motion_method=storm_motion_method,
    )
    SRH = srh_result["SRH_m2_s2"]

    # EHI convention: use absolute SRH value — negative SRH (anticyclonic)
    # environments get EHI = 0 since they don't support right-moving supercells
    EHI = float(max(CAPE * SRH, 0.0) / 160_000.0)

    return {
        "EHI": EHI,
    }


def supercell_composite(
        CAPE: float,  # J/kg  (from cape_cin())
        z: np.ndarray,  # altitude profile [m]
        u_ms: np.ndarray,  # zonal wind profile [m/s]
        v_ms: np.ndarray,  # meridional wind profile [m/s]
        z_srh_top: float = 3000.0,  # SRH integration depth [m]
        z_bwd_top: float = 6000.0,  # BWD upper level [m]
        u_storm: float = None,
        v_storm: float = None,
        storm_motion_method: str = "bunkers",
):
    """
    Supercell Composite Parameter.

        SCP = (CAPE / 1000)  ×  (SRH / 50)  ×  (BWD / 20)

    All three terms are normalised by their "minimum threshold" values
    so SCP = 1 when CAPE = 1000 J/kg, SRH = 50 m²/s², BWD = 20 m/s —
    representing a just-barely-supercell-capable environment.

    Conditional zeroing (Thompson et al. 2003):
      SCP = 0 if CAPE < 100 J/kg (no meaningful buoyancy)
      SCP = 0 if SRH  < 50 m²/s² (no meaningful rotation)
      SCP = 0 if BWD  < 10 m/s   (insufficient deep-layer shear)

    Horizontal neighbours needed : 0
    Vertical neighbours needed   : ≥5 UAVs spanning 0 → 6 km
                                   (covers both SRH 0-3 km and BWD 0-6 km)

    Parameters
    ----------
    CAPE      : J/kg
    z, u_ms, v_ms : wind profile
    z_srh_top : SRH integration depth [m]  default 3000
    z_bwd_top : BWD upper level [m]        default 6000
    """
    srh_result = storm_relative_helicity(
        z, u_ms, v_ms,
        z_top=z_srh_top,
        u_storm=u_storm,
        v_storm=v_storm,
        storm_motion_method=storm_motion_method,
    )
    bwd_result = bulk_wind_difference(z, u_ms, v_ms, z_top=z_bwd_top)

    SRH = srh_result["SRH_m2_s2"]
    BWD = bwd_result["BWD_ms"]

    # ── conditional zeroing ──
    cape_term = CAPE / 1000.0
    srh_term = SRH / 50.0
    bwd_term = BWD / 20.0

    if CAPE < 100.0:
        cape_term = 0.0
    if SRH < 50.0:
        srh_term = 0.0
    if BWD < 10.0:
        bwd_term = 0.0

    SCP = float(cape_term * srh_term * bwd_term)

    return {
        "SCP": SCP,
    }