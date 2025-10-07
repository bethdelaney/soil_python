import os
import math
from typing import Optional
import numpy as np
import pandas as pd

def gaus_convolve_to_landsat8(
    lab_data: pd.DataFrame,
    reflectance_prefix: str = "spc.",
    center_csv_path: Optional[str] = None,         # leave None to auto-find in same folder as this file
    center_csv_filename: str = "l8_wavelengths.csv",
    name_col: str = "Name",
    center_col: str = "Wavelength",
    fwhm_col: Optional[str] = "FWHM",              # set None if your CSV lacks this column
    fallback_fwhm: float = 10.0,                   # used only if per-band FWHM missing/NaN
    min_coverage: float = 0.80                     # require ≥ this fraction of SRF mass inside lab range
) -> pd.DataFrame:
    """
    Convolve lab spectra to Landsat-8 bands using *Gaussian SRFs only*.

    CSV is expected to be in the same folder as this file by default (center_csv_filename).
    If you pass center_csv_path it will use that explicit path instead.

    Returns: original metadata + one column per convolved band named "<Name> - <centre_nm:0.2f>".
    """

    # ---------- Find CSV (prefer explicit path; else same folder as this file; else CWD) ----------
    if center_csv_path and os.path.exists(center_csv_path):
        srf_file_path = center_csv_path
    else:
        package_dir = os.path.dirname(__file__)
        candidate = os.path.join(package_dir, center_csv_filename)
        if os.path.exists(candidate):
            srf_file_path = candidate
        elif os.path.exists(center_csv_filename):
            srf_file_path = os.path.abspath(center_csv_filename)
        else:
            raise FileNotFoundError(
                f"Band-centre CSV not found. Tried: "
                f"{center_csv_path!r}, {candidate!r}, {os.path.abspath(center_csv_filename)!r}"
            )

    bands = pd.read_csv(srf_file_path)
    if name_col not in bands.columns or center_col not in bands.columns:
        raise ValueError(f"CSV must contain '{name_col}' and '{center_col}' columns.")

    keep_cols = [name_col, center_col] + ([fwhm_col] if (fwhm_col and fwhm_col in bands.columns) else [])
    bands = bands[keep_cols].dropna(subset=[name_col, center_col]).copy()
    bands[name_col] = bands[name_col].astype(str)
    bands[center_col] = bands[center_col].astype(float)
    bands.sort_values(center_col, inplace=True)

    # ---------- Split metadata vs reflectances ----------
    refl_cols = [c for c in lab_data.columns if c.startswith(reflectance_prefix)]
    if not refl_cols:
        raise ValueError(f"No reflectance columns found with prefix '{reflectance_prefix}'.")
    meta_cols = [c for c in lab_data.columns if c not in refl_cols]
    metadata = lab_data[meta_cols].copy()

    # Lab wavelengths & reflectance matrix, sorted by wavelength
    try:
        lab_wls = np.array([float(c.replace(reflectance_prefix, "")) for c in refl_cols], dtype=float)
    except Exception as e:
        raise ValueError(
            f"Could not parse wavelengths from column names. "
            f"Expected '{reflectance_prefix}<wavelength>' like 'spc.400'. Error: {e}"
        )
    order = np.argsort(lab_wls)
    lab_wls = lab_wls[order]
    R = lab_data[refl_cols].values[:, order]

    wl_min, wl_max = lab_wls[0], lab_wls[-1]

    # ---------- Helpers ----------
    def fwhm_to_sigma(fwhm: float) -> float:
        return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def gaussian_coverage(center: float, sigma: float) -> float:
        # Fraction of Gaussian mass within [wl_min, wl_max] via normal CDF
        if sigma <= 0:
            return 0.0
        def Phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        zmin = (wl_min - center) / sigma
        zmax = (wl_max - center) / sigma
        return max(0.0, min(1.0, Phi(zmax) - Phi(zmin)))

    # ---------- Convolution (Gaussian only) ----------
    out = {}
    for _, row in bands.iterrows():
        center = float(row[center_col])
        if fwhm_col and fwhm_col in bands.columns and pd.notna(row.get(fwhm_col)):
            fwhm = float(row[fwhm_col])
        else:
            fwhm = float(fallback_fwhm)

        if fwhm <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        sigma = fwhm_to_sigma(fwhm)

        # Require adequate overlap with lab spectrum
        if gaussian_coverage(center, sigma) < min_coverage:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        # Gaussian SRF on lab grid; normalise to unit area
        S = np.exp(-0.5 * ((lab_wls - center) / sigma) ** 2)
        area = np.trapz(S, lab_wls)
        if area <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue
        W = S / area  # ∫ W(λ) dλ = 1 on the lab grid

        # Band value for each sample: ∫ R(λ) W(λ) dλ
        band_vals = np.trapz(R * W[np.newaxis, :], lab_wls, axis=1)
        out[f"{row[name_col]} - {center:.2f}"] = band_vals

    return pd.concat([metadata, pd.DataFrame(out, index=lab_data.index)], axis=1)

# l8 = convolve_to_landsat8(
#     df, 
#     reflectance_prefix="spc.", 
#     center_csv_path=None,                # auto-find
#     center_csv_filename="l8_wavelengths.csv",  # same folder as this .py file
#     fwhm_col="FWHM",                     # if your CSV has it (yours does)
#     fallback_fwhm=10.0                   # used only if FWHM missing for a band
# )

def gaus_convolve_to_landsatnext(
    lab_data: pd.DataFrame,
    reflectance_prefix: str = "spc.",
    center_csv_path: Optional[str] = None,          # leave None to auto-find in same folder as this file
    center_csv_filename: str = "landsatnext_wavelengths.csv",
    name_col: str = "Name",
    center_col: str = "Wavelength",
    fwhm_col: Optional[str] = "FWHM",               # set None if your CSV lacks this column
    fallback_fwhm: float = 10.0,                    # used only if per-band FWHM missing/NaN
    min_coverage: float = 0.80                      # require ≥ this fraction of SRF mass inside lab range
) -> pd.DataFrame:
    """
    Convolve lab spectra to Landsat-Next bands using *Gaussian SRFs only*.

    CSV is expected in the same folder as this file by default (center_csv_filename).
    If you pass center_csv_path, it will use that explicit path instead.

    Returns: original metadata + one column per convolved band named "<Name> - <centre_nm:0.2f>".
    """

    # ---------- Locate CSV (prefer explicit path; else same folder; else CWD) ----------
    if center_csv_path and os.path.exists(center_csv_path):
        srf_file_path = center_csv_path
    else:
        package_dir = os.path.dirname(__file__)
        candidate = os.path.join(package_dir, center_csv_filename)
        if os.path.exists(candidate):
            srf_file_path = candidate
        elif os.path.exists(center_csv_filename):
            srf_file_path = os.path.abspath(center_csv_filename)
        else:
            raise FileNotFoundError(
                f"Band-centre CSV not found. Tried: "
                f"{center_csv_path!r}, {candidate!r}, {os.path.abspath(center_csv_filename)!r}"
            )

    bands = pd.read_csv(srf_file_path)
    if name_col not in bands.columns or center_col not in bands.columns:
        raise ValueError(f"CSV must contain '{name_col}' and '{center_col}' columns.")
    keep_cols = [name_col, center_col] + ([fwhm_col] if (fwhm_col and fwhm_col in bands.columns) else [])
    bands = bands[keep_cols].dropna(subset=[name_col, center_col]).copy()
    bands[name_col] = bands[name_col].astype(str)
    bands[center_col] = bands[center_col].astype(float)
    bands.sort_values(center_col, inplace=True)

    # ---------- Split metadata vs reflectances ----------
    refl_cols = [c for c in lab_data.columns if c.startswith(reflectance_prefix)]
    if not refl_cols:
        raise ValueError(f"No reflectance columns found with prefix '{reflectance_prefix}'.")
    meta_cols = [c for c in lab_data.columns if c not in refl_cols]
    metadata = lab_data[meta_cols].copy()

    # Lab wavelengths & reflectance matrix, sorted by wavelength
    try:
        lab_wls = np.array([float(c.replace(reflectance_prefix, "")) for c in refl_cols], dtype=float)
    except Exception as e:
        raise ValueError(
            f"Could not parse wavelengths from column names. "
            f"Expected '{reflectance_prefix}<wavelength>' like 'spc.400'. Error: {e}"
        )
    order = np.argsort(lab_wls)
    lab_wls = lab_wls[order]
    R = lab_data[refl_cols].values[:, order]

    wl_min, wl_max = lab_wls[0], lab_wls[-1]

    # ---------- Helpers ----------
    def fwhm_to_sigma(fwhm: float) -> float:
        return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def gaussian_coverage(center: float, sigma: float) -> float:
        # Fraction of Gaussian mass within [wl_min, wl_max] via normal CDF
        if sigma <= 0:
            return 0.0
        def Phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        zmin = (wl_min - center) / sigma
        zmax = (wl_max - center) / sigma
        return max(0.0, min(1.0, Phi(zmax) - Phi(zmin)))

    # ---------- Convolution (Gaussian only) ----------
    out = {}
    for _, row in bands.iterrows():
        center = float(row[center_col])
        if fwhm_col and fwhm_col in bands.columns and pd.notna(row.get(fwhm_col)):
            fwhm = float(row[fwhm_col])
        else:
            fwhm = float(fallback_fwhm)

        if fwhm <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        sigma = fwhm_to_sigma(fwhm)

        # Require adequate overlap with lab spectrum
        if gaussian_coverage(center, sigma) < min_coverage:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        # Gaussian SRF on lab grid; normalise to unit area
        S = np.exp(-0.5 * ((lab_wls - center) / sigma) ** 2)
        area = np.trapz(S, lab_wls)
        if area <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue
        W = S / area  # ∫ W(λ) dλ = 1 on the lab grid

        # Band value for each sample: ∫ R(λ) W(λ) dλ
        band_vals = np.trapz(R * W[np.newaxis, :], lab_wls, axis=1)
        out[f"{row[name_col]} - {center:.2f}"] = band_vals

    return pd.concat([metadata, pd.DataFrame(out, index=lab_data.index)], axis=1)

# lnext = convolve_to_landsatnext(
#     df,
#     reflectance_prefix="spc.",
#     center_csv_filename="landsatnext_wavelengths.csv",  # same folder as this .py
#     fwhm_col="FWHM",                                   # if present in your CSV
#     fallback_fwhm=10.0
# )

def gaus_convolve_to_enmap(
    lab_data: pd.DataFrame,
    reflectance_prefix: str = "spc.",
    center_csv_path: Optional[str] = None,          # leave None to auto-find in same folder as this file
    center_csv_filename: str = "enmap_wavelengths.csv",
    name_col: str = "Name",
    center_col: str = "Wavelength",
    fwhm_col: Optional[str] = "FWHM",               # set None if your CSV lacks this column
    fallback_fwhm: float = 10.0,                    # used only if per-band FWHM missing/NaN
    min_coverage: float = 0.80                      # require ≥ this fraction of SRF mass inside lab range
) -> pd.DataFrame:
    """
    Convolve lab spectra to EnMAP bands using *Gaussian SRFs only*.

    The CSV is expected in the same folder as this file by default (center_csv_filename).
    If you pass center_csv_path, it will use that explicit path instead.

    Returns: original metadata + one column per convolved band named "<Name> - <centre_nm:0.2f>".
    """

    # ---------- Locate CSV (prefer explicit path; else same folder; else CWD) ----------
    if center_csv_path and os.path.exists(center_csv_path):
        srf_file_path = center_csv_path
    else:
        package_dir = os.path.dirname(__file__)
        candidate = os.path.join(package_dir, center_csv_filename)
        if os.path.exists(candidate):
            srf_file_path = candidate
        elif os.path.exists(center_csv_filename):
            srf_file_path = os.path.abspath(center_csv_filename)
        else:
            raise FileNotFoundError(
                f"Band-centre CSV not found. Tried: "
                f"{center_csv_path!r}, {candidate!r}, {os.path.abspath(center_csv_filename)!r}"
            )

    bands = pd.read_csv(srf_file_path)
    if name_col not in bands.columns or center_col not in bands.columns:
        raise ValueError(f"CSV must contain '{name_col}' and '{center_col}' columns.")
    keep_cols = [name_col, center_col] + ([fwhm_col] if (fwhm_col and fwhm_col in bands.columns) else [])
    bands = bands[keep_cols].dropna(subset=[name_col, center_col]).copy()
    bands[name_col] = bands[name_col].astype(str)
    bands[center_col] = bands[center_col].astype(float)
    bands.sort_values(center_col, inplace=True)

    # ---------- Split metadata vs reflectances ----------
    refl_cols = [c for c in lab_data.columns if c.startswith(reflectance_prefix)]
    if not refl_cols:
        raise ValueError(f"No reflectance columns found with prefix '{reflectance_prefix}'.")
    meta_cols = [c for c in lab_data.columns if c not in refl_cols]
    metadata = lab_data[meta_cols].copy()

    # Lab wavelengths & reflectance matrix, sorted by wavelength
    try:
        lab_wls = np.array([float(c.replace(reflectance_prefix, "")) for c in refl_cols], dtype=float)
    except Exception as e:
        raise ValueError(
            f"Could not parse wavelengths from column names. "
            f"Expected '{reflectance_prefix}<wavelength>' like 'spc.400'. Error: {e}"
        )
    order = np.argsort(lab_wls)
    lab_wls = lab_wls[order]
    R = lab_data[refl_cols].values[:, order]

    wl_min, wl_max = lab_wls[0], lab_wls[-1]

    # ---------- Helpers ----------
    def fwhm_to_sigma(fwhm: float) -> float:
        return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def gaussian_coverage(center: float, sigma: float) -> float:
        # Fraction of Gaussian mass within [wl_min, wl_max] via normal CDF
        if sigma <= 0:
            return 0.0
        def Phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        zmin = (wl_min - center) / sigma
        zmax = (wl_max - center) / sigma
        return max(0.0, min(1.0, Phi(zmax) - Phi(zmin)))

    # ---------- Convolution (Gaussian only) ----------
    out = {}
    for _, row in bands.iterrows():
        center = float(row[center_col])
        if fwhm_col and fwhm_col in bands.columns and pd.notna(row.get(fwhm_col)):
            fwhm = float(row[fwhm_col])
        else:
            fwhm = float(fallback_fwhm)

        if fwhm <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        sigma = fwhm_to_sigma(fwhm)

        # Require adequate overlap with lab spectrum
        if gaussian_coverage(center, sigma) < min_coverage:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue

        # Gaussian SRF on lab grid; normalise to unit area
        S = np.exp(-0.5 * ((lab_wls - center) / sigma) ** 2)
        area = np.trapz(S, lab_wls)
        if area <= 0:
            out[f"{row[name_col]} - {center:.2f}"] = np.full(R.shape[0], np.nan)
            continue
        W = S / area  # ∫ W(λ) dλ = 1 on the lab grid

        # Band value for each sample: ∫ R(λ) W(λ) dλ
        band_vals = np.trapz(R * W[np.newaxis, :], lab_wls, axis=1)
        out[f"{row[name_col]} - {center:.2f}"] = band_vals

    return pd.concat([metadata, pd.DataFrame(out, index=lab_data.index)], axis=1)

# enmap = convolve_to_enmap(
#     df,
#     reflectance_prefix="spc.",
#     center_csv_filename="enmap_wavelengths.csv",  # same folder as this .py
#     fwhm_col="FWHM",                              # if present in your CSV
#     fallback_fwhm=10.0
# )
