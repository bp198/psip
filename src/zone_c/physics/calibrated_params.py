"""
PHMSA-Calibrated Distribution Parameters
=========================================
Auto-generated from phmsa_eda_calibration.py
Source: PHMSA Form F 7100.2 — Gas Transmission Incidents 2010-2025
Records: 1996 total incidents, 216 material/weld failures
Calibration date: 2026-03-24 00:06

Reference pipe: D=323.8 mm, B=7.11 mm (PHMSA median values)
"""

from ..physics.mc_failure_probability import DistributionParams


# === Defect Size Distributions ===
# Depth: calibrated from PHMSA wall thickness at failure + ILI literature
# Median a = 0.20 * B_median = 1.43 mm
DIST_DEFECT_A = DistributionParams(
    dist_type="lognormal", param1=0.3559, param2=0.75,
    lower_bound=0.3, upper_bound=6.8,
)  # Median=1.43 mm, Mean=1.89 mm

# Length: aspect ratio a/c ~ 0.1-0.5 (ILI data), median 2c ~ 6 * median a
DIST_DEFECT_2C = DistributionParams(
    dist_type="lognormal", param1=2.1477, param2=0.65,
    lower_bound=1.0, upper_bound=200.0,
)  # Median=8.56 mm, Mean=10.58 mm

# === Material Property Distributions ===
# Fracture toughness: Weibull, conservative for vintage pipe (median age ~61 yrs)
DIST_K_MAT = DistributionParams(
    dist_type="weibull", param1=3.5, param2=110.0,
    lower_bound=25.0,
)  # Mean~99.2 MPa*sqrt(m)

# Yield strength: empirical from PHMSA PIPE_SMYS (N=900)
# Population dominated by X52 (358 MPa), X42 (290 MPa), X60 (414 MPa)
DIST_SIGMA_Y = DistributionParams(
    dist_type="normal", param1=316.9, param2=78.4,
    lower_bound=200.0,
)  # Empirical mean=316.9 MPa

# Ultimate tensile strength: estimated as 1.2 * SMYS
DIST_SIGMA_U = DistributionParams(
    dist_type="normal", param1=380.2, param2=86.2,
    lower_bound=301.0,
)  # Estimated from API 5L Y/T ratio

# === Operating Conditions ===
# Pressure: Barlow MAOP estimate, Class 1 (F=0.72), from PHMSA pipe geometry (N=877)
DIST_PRESSURE_CLASS1 = DistributionParams(
    dist_type="normal", param1=10.25, param2=4.48,
    lower_bound=0.5,
)  # Estimated MAOP for PHMSA median pipe

# === SCF by Weld / Seam Type ===
# Calibrated from PHMSA seam type failure rates + IIW FAT class mapping
SCF_DISTRIBUTIONS = {
    "seamless": DistributionParams(
        dist_type="uniform", param1=1.0, param2=1.2,
    ),  # FAT 125+, lowest vulnerability
    "butt_weld_ground_flush": DistributionParams(
        dist_type="uniform", param1=1.0, param2=1.3,
    ),  # FAT 112
    "dsaw_seam": DistributionParams(
        dist_type="uniform", param1=1.1, param2=1.5,
    ),  # FAT 90, DSAW
    "erw_hf_seam": DistributionParams(
        dist_type="uniform", param1=1.1, param2=1.6,
    ),  # FAT 90, ERW high frequency
    "butt_weld_as_welded": DistributionParams(
        dist_type="uniform", param1=1.2, param2=2.0,
    ),  # FAT 90, as-welded
    "girth_weld_field": DistributionParams(
        dist_type="uniform", param1=1.3, param2=2.5,
    ),  # FAT 71, field girth welds
    "erw_lf_seam": DistributionParams(
        dist_type="uniform", param1=1.3, param2=2.5,
    ),  # FAT 71, ERW low frequency (high vulnerability)
    "fillet_weld_branch": DistributionParams(
        dist_type="uniform", param1=1.5, param2=3.5,
    ),  # FAT 63
    "lap_welded": DistributionParams(
        dist_type="uniform", param1=1.5, param2=3.0,
    ),  # FAT 63, lap welded vintage pipe
    "socket_weld": DistributionParams(
        dist_type="uniform", param1=2.0, param2=4.0,
    ),  # FAT 56
}

# === PHMSA Historical Reference Statistics ===
PHMSA_TOTAL_INCIDENTS_2010_2025 = 1996
PHMSA_MATERIAL_WELD_FAILURES = 216
PHMSA_MEAN_INCIDENTS_PER_YEAR = 117.4
PHMSA_MEAN_MAT_WELD_PER_YEAR = 12.7
PHMSA_TRANSMISSION_MILEAGE = 305_000  # miles (approximate)
PHMSA_WELD_FAILURE_RATE_PER_1000MI_YR = round(
    12.7 / (305_000 / 1000), 4
)  # = 0.0417 per 1000 mi-yr
PHMSA_MEDIAN_PIPE_AGE_AT_FAILURE = 61.0
