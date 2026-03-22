"""
Estimate Eddy Dissipation Rate (EDR) from a single UAV's
onboard wind sensor time-series.

EDR is the ICAO standard turbulence intensity metric [m^(2/3) s^-1].
It is the cube root of the turbulent kinetic energy dissipation rate ε:

    EDR = ε^(1/3)

Three independent estimation methods are provided.  Use all three
and take the median for robustness.

Method 1 — Structure Function (SF), Better for short, non-stationary bursts.
────────────────────────────────────
    D_L(r) = <[u(x+r) - u(x)]^2> = C² · ε^(2/3) · r^(2/3)

u(x) is wind velocity at a specific point.
r is The distance (separation) between two measurements.
<> is The "expected value" or average over many samples.
C² ≈ 2.0 is the Kolmogorov constant for the longitudinal structure function.
log D_L(r) = (2/3) log(ε) + log(C²) + (2/3) log(r)
log D_L(r) and (2/3) log(r) are sensor readings
m(slope) = 2/3
b(intercept) = log(C · ε^(2/3))


Method 2 — Power Spectral Density (PSD), Better for continuous, steady turbulence.
─────────────────────────────────────────
    S(f) = A · (U/2π)^(2/3) · ε^(2/3) · f^(-5/3)

looks at the energy of the wind in the frequency domain.
It is based on the principle that turbulence is a collection of waves (eddies) of different sizes.
Fast, small waves have less energy than slow, large waves.

S(f): Power Spectral Density.
f is frequency in Hz, We record wind data and perform a Fast Fourier Transform (FFT)
Fit a -5/3 slope to the log-log PSD and extract ε^(2/3) from the
intercept.  A = 0.55 is the one-dimensional Kolmogorov constant.

Method 3 — Variance Method (fast, less accurate)
──────────────────────────────────────────────────
Quick approximation from the standard deviation of vertical wind:

    ε ≈ (σ_w / C_w)^3 / L

The method assumes that the variance of the wind velocity σ is proportional to the Eddy Dissipation Rate ε and the scale of the turbulence L
    where C_w ≈ 1.9 and L is The "Integral Length Scale" (the size of the largest eddies). (~100 m).
Useful when only a short burst of samples is available.

Sampling requirements
──────────────────────
  Minimum sampling rate : 5 Hz
  Recommended           : 10–25 Hz
  Minimum window length : 10 s   (50–250 samples)
  Recommended window    : 30–60 s

Rotor correction
─────────────────
Multi-rotor UAVs induce a downwash bias in w_ms.  A high-pass
filter (cutoff ~0.5 Hz) is applied before all calculations to
remove the DC rotor signature.  The filter cutoff is configurable.

Units
─────
  Input  : wind components in m/s, sampling rate in Hz
  Output : EDR in m^(2/3) s^-1

"""

import numpy as np
from scipy import signal
from dataclasses import dataclass, field

EDR_THRESHOLDS = {
    "calm":                 (0.00, 0.05),
    "light":                (0.05, 0.10),
    "light-to-moderate":    (0.10, 0.22),
    "moderate":             (0.22, 0.34),
    "moderate-to-severe":   (0.34, 0.45),
    "severe":               (0.45, np.inf),
}

# Kolmogorov constants
C2_LONGITUDINAL  = 2.0    # 2nd-order longitudinal structure function
A_1D             = 0.55   # 1-D longitudinal PSD prefactor
C_W              = 1.9    # vertical variance method constant
L_OUTER_M        = 100.0  # assumed outer length scale [m]


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class EDRResult:
    edr_sf:            float       # structure function estimate
    edr_psd:           float       # PSD estimate
    edr_var:           float       # variance method estimate
    edr_median:        float       # robust median of three methods

    confidence:        str         # "high" / "medium" / "low"
    n_samples:         int
    sampling_rate_hz:  float
    window_s:          float       # window length in seconds

    sf_slope:          float       # fitted slope (should be ~0.667)
    psd_slope:         float       # fitted slope (should be ~-1.667)

    warnings:          list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────
def _highpass_filter(signal: np.ndarray, fs: float,
                     cutoff_hz: float = 0.5) -> np.ndarray:
    """
    Simple high-pass filter via subtraction of a moving average.
    Removes low-frequency platform motion and rotor DC bias.

    cutoff_hz ≈ 0.5 removes variations slower than 2 seconds,
    which covers rotor-induced downwash and flight path curvature.
    """
    window = max(3, int(fs / cutoff_hz))
    kernel = np.ones(window) / window
    trend  = np.convolve(signal, kernel, mode="same")
    return signal - trend


def _confidence(edr_sf, edr_psd, edr_var, n_samples, sf_slope, psd_slope):
    """
    Estimate confidence in the EDR measurement.

    High   : all three methods agree within 30%, slopes near theoretical
    Medium : two methods agree, sample count adequate
    Low    : large spread between methods or poor spectral slope
    """
    vals   = np.array([edr_sf, edr_psd, edr_var])
    spread = np.std(vals) / (np.mean(vals) + 1e-9)

    slope_ok   = abs(sf_slope - 2/3) < 0.2 and abs(psd_slope + 5/3) < 0.3
    enough_pts = n_samples >= 100

    if spread < 0.30 and slope_ok and enough_pts:
        return "high"
    elif spread < 0.60 and enough_pts:
        return "medium"
    else:
        return "low"


# ── Method 1: Structure Function ──────────────────────────────

def _edr_structure_function(
    u_fluct: np.ndarray,
    dx_m:    float,
    min_lags: int = 5,
    max_lags: int = 30,
) -> tuple[float, float]:
    """
    Estimate EDR from the 2nd-order longitudinal structure function.

    D_L(r) = C² · ε^(2/3) · r^(2/3)

    Parameters
    ----------
    u_fluct  : longitudinal velocity fluctuations [m/s], high-pass filtered
    dx_m     : spatial step between consecutive samples [m]
               = airspeed [m/s] / sampling_rate [Hz]  (Taylor frozen turbulence)
    min_lags : minimum lag index to include in fit (avoid noise floor)
    max_lags : maximum lag index to include in fit (stay in inertial subrange)
    """
    n       = len(u_fluct)
    max_lag = min(max_lags, n // 4)
    lags    = np.arange(min_lags, max_lag + 1)

    sf = np.array([
        np.mean((u_fluct[lag:] - u_fluct[:-lag])**2)
        for lag in lags
    ])

    r = lags * dx_m          # physical separation [m]

    # fit log D_L = (2/3) log(ε) + log(C²) + (2/3) log(r)
    # i.e. log(sf) = A + slope * log(r)
    log_r  = np.log(r)
    log_sf = np.log(sf + 1e-20)

    coeffs = np.polyfit(log_r, log_sf, 1)
    slope  = coeffs[0]
    intercept = coeffs[1]

    # extract ε^(2/3) = exp(intercept) / C²
    eps_23 = np.exp(intercept) / C2_LONGITUDINAL
    edr    = max(eps_23 ** (3/2), 0.0) ** (1/3)

    return float(edr), float(slope)


# ── Method 2: Power Spectral Density ──────────────────────────

def _edr_psd(
    u_fluct:   np.ndarray,
    fs:        float,
    airspeed:  float,
    f_low_hz:  float = 0.5,
    f_high_hz: float = None,
) -> tuple[float, float]:
    """
    Estimate EDR from the -5/3 inertial subrange of the velocity PSD.

    S(f) = A · (U / 2π)^(2/3) · ε^(2/3) · f^(-5/3)

    Uses Welch's method for a stable PSD estimate.

    Parameters
    ----------
    u_fluct   : longitudinal velocity fluctuations [m/s]
    fs        : sampling rate [Hz]
    airspeed  : mean horizontal airspeed [m/s]  (Taylor hypothesis)
    f_low_hz  : lower bound of inertial subrange fit [Hz]
    f_high_hz : upper bound; defaults to fs/4 (below aliasing)
    """
    if f_high_hz is None:
        f_high_hz = fs / 4.0

    n        = len(u_fluct)
    seg_len  = min(256, n // 4)

    # Welch PSD
    # freqs, psd = _welch(u_fluct, fs, seg_len)
    freqs, psd = signal.welch(
        u_fluct,
        fs=fs,
        window='hann',  # Same as np.hanning
        nperseg=seg_len,  # Same as seg_len
        noverlap=seg_len // 2,  # Same as hop
        scaling='density',  # Ensures units are m^2/s^2 / Hz
        detrend='constant'  # Same as dropping the [0] (DC) component
    )
    # fit in inertial subrange
    mask = (freqs >= f_low_hz) & (freqs <= f_high_hz) & (psd > 0)
    if mask.sum() < 4:
        return 0.0, -5/3   # not enough points for fit

    log_f   = np.log(freqs[mask])
    log_psd = np.log(psd[mask])

    coeffs    = np.polyfit(log_f, log_psd, 1)
    slope     = coeffs[0]        # should be ~-5/3 ≈ -1.667
    intercept = coeffs[1]

    # S(f) = A · (U/2π)^(2/3) · ε^(2/3) · f^(-5/3)
    # at reference f=1: exp(intercept) = A · (U/2π)^(2/3) · ε^(2/3)
    U_term   = (airspeed / (2 * np.pi)) ** (2/3)
    eps_23   = np.exp(intercept) / (A_1D * U_term + 1e-20)
    edr      = max(eps_23, 0.0) ** (3/4)   # eps^(2/3) -> eps -> edr=eps^(1/3)

    return float(edr), float(slope)


def _welch(x: np.ndarray, fs: float, seg_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch power spectral density estimate using Hanning windows.
    """
    n       = len(x)
    hop     = seg_len // 2
    window  = np.hanning(seg_len)
    w_norm  = np.sum(window**2)

    n_segs  = max(1, (n - seg_len) // hop + 1)
    psd_sum = np.zeros(seg_len // 2 + 1)

    for k in range(n_segs):
        start  = k * hop
        seg    = x[start:start + seg_len]
        if len(seg) < seg_len:
            break
        ft      = np.fft.rfft(seg * window)
        psd_seg = (np.abs(ft)**2) / (fs * w_norm)
        psd_seg[1:-1] *= 2    # one-sided spectrum
        psd_sum += psd_seg

    freqs = np.fft.rfftfreq(seg_len, d=1.0/fs)
    psd   = psd_sum / n_segs

    return freqs[1:], psd[1:]    # drop DC component


# ── Method 3: Variance Method ─────────────────────────────────

def _edr_variance(w_fluct: np.ndarray) -> float:
    """
    Fast EDR estimate from vertical wind variance.

        ε ≈ (σ_w / C_w)^3 / L_outer

    Very quick but less accurate — assumes a specific outer scale L.
    Good as a sanity check or when sample count is small.
    """
    sigma_w = float(np.std(w_fluct))
    eps     = (sigma_w / C_W)**3 / L_OUTER_M
    return max(eps, 0.0) ** (1/3)


# ── Main EDR estimator ────────────────────────────────────────

def estimate_edr(
    u_ms:        np.ndarray,     # zonal wind time series [m/s]
    v_ms:        np.ndarray,     # meridional wind time series [m/s]
    w_ms:        np.ndarray,     # vertical wind time series [m/s]
    fs_hz:       float,          # sampling rate [Hz]
    hp_cutoff:   float = 0.5,    # high-pass filter cutoff [Hz]
    sf_min_lag:  int   = 3,
    sf_max_lag:  int   = 40,
) -> EDRResult:
    """
    Parameters
    ----------
    u_ms        : east-west wind [m/s], shape (N,)
    v_ms        : north-south wind [m/s], shape (N,)
    w_ms        : vertical wind [m/s], shape (N,)
    fs_hz       : sampling frequency [Hz]
    hp_cutoff   : high-pass filter cutoff [Hz] (removes rotor bias)
    sf_min_lag  : minimum lag for structure function fit
    sf_max_lag  : maximum lag for structure function fit
    """
    u   = np.asarray(u_ms, dtype=float)
    v   = np.asarray(v_ms, dtype=float)
    w   = np.asarray(w_ms, dtype=float)
    n   = len(u)

    warnings = []

    if n < int(fs_hz * 5):
        warnings.append(
            f"Only {n/fs_hz:.1f}s of data — recommend >= 10s for reliable EDR."
        )
    if fs_hz < 5:
        warnings.append(
            f"Sampling rate {fs_hz} Hz is below recommended 5 Hz minimum."
        )

    # ── high-pass filter: remove rotor downwash and platform motion ──
    u_f = _highpass_filter(u, fs_hz, cutoff_hz=hp_cutoff)
    v_f = _highpass_filter(v, fs_hz, cutoff_hz=hp_cutoff)
    w_f = _highpass_filter(w, fs_hz, cutoff_hz=hp_cutoff)

    # ── mean airspeed (for Taylor frozen turbulence hypothesis) ──
    airspeed = float(np.mean(np.sqrt(u**2 + v**2)))
    if airspeed < 1.0:
        warnings.append(
            "Airspeed < 1 m/s — Taylor hypothesis may not hold. "
            "PSD and SF estimates less reliable for hovering UAVs."
        )
        airspeed = max(airspeed, 1.0)

    # spatial step via Taylor: dx = U / fs
    dx_m = airspeed / fs_hz

    # ── Method 1: structure function on longitudinal component ──
    # longitudinal = projection onto mean wind direction
    u_mean = float(np.mean(u_f))
    v_mean = float(np.mean(v_f))
    spd_mean = np.sqrt(u_mean**2 + v_mean**2) + 1e-9
    u_long = u_f * (u_mean / spd_mean) + v_f * (v_mean / spd_mean)

    edr_sf, sf_slope = _edr_structure_function(
        u_long, dx_m, min_lags=sf_min_lag, max_lags=sf_max_lag
    )

    if abs(sf_slope - 2/3) > 0.4:
        warnings.append(
            f"Structure function slope={sf_slope:.2f} deviates from "
            "Kolmogorov 2/3 — turbulence may not be fully developed."
        )

    # ── Method 2: PSD on longitudinal component ──
    edr_psd, psd_slope = _edr_psd(u_long, fs_hz, airspeed)

    if abs(psd_slope + 5/3) > 0.5:
        warnings.append(
            f"PSD slope={psd_slope:.2f} deviates from -5/3 law."
        )

    # ── Method 3: variance method on vertical wind ──
    edr_var = _edr_variance(w_f)

    # ── robust median of three estimates ──
    edr_median = float(np.median([edr_sf, edr_psd, edr_var]))

    # cap physically unreasonable values
    edr_sf     = float(np.clip(edr_sf,     0.0, 2.0))
    edr_psd    = float(np.clip(edr_psd,    0.0, 2.0))
    edr_var    = float(np.clip(edr_var,    0.0, 2.0))
    edr_median = float(np.clip(edr_median, 0.0, 2.0))

    confidence = _confidence(
        edr_sf, edr_psd, edr_var, n, sf_slope, psd_slope
    )

    return EDRResult(
        edr_sf           = edr_sf,
        edr_psd          = edr_psd,
        edr_var          = edr_var,
        edr_median       = edr_median,
        confidence       = confidence,
        n_samples        = n,
        sampling_rate_hz = fs_hz,
        window_s         = n / fs_hz,
        sf_slope         = sf_slope,
        psd_slope        = psd_slope,
        warnings         = warnings,
    )

