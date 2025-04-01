import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial

# Citations:
# 1. Buddenbaum, H. and Steffens, M., 2012. The effects of spectral pretreatments on chemometric analyses of soil profiles using laboratory imaging spectroscopy. Applied and Environmental Soil Science, 2012(1), p.274903.
# https://doi.org/10.1155/2012/274903
# 2. Martens, H. and Stark, E., 1991. Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy. Journal of pharmaceutical and biomedical analysis, 9(8), pp.625-635.
# https://doi.org/10.1016/0731-7085(91)80188-F

def snv_detrend(X):
    """
    Apply Standard Normal Variate (SNV) and 2nd order polynomial detrending to spectral data.
    
    Parameters:
    - X: 2D numpy array or pandas DataFrame (rows = wavelengths, columns = samples)

    Returns:
    - SNV-DT transformed spectra as numpy array
    """
    snv_data = (df - df.mean(axis=0)) / df.std(axis=0)
    detrended = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    wavelengths = np.arange(len(df.index))  # or use df.index if numeric

    for col in df.columns:
        coeffs = Polynomial.fit(wavelengths, snv_data[col].values, deg=2).convert().coef
        baseline = coeffs[0] + coeffs[1] * wavelengths + coeffs[2] * wavelengths**2
        detrended[col] = snv_data[col].values - baseline

    return detrended

# Example usage:
# snv_detrended_spectra = snv_detrend(df)
# snv_detrended_df.insert(0, "wavelength", df.index) # Add wavelengths back as a column


def continuum_removal(df, wavelengths=None, normalise=True):
    """
    Perform Continuum Removal and optional Band Depth Normalization on spectra.
    
    Parameters:
    - df: DataFrame with wavelength rows and sample columns
    - wavelengths: optional, defaults to df.index if not provided
    - normalize: if True, scales CR spectra to [0, 1] range (BDN)

    Returns:
    - DataFrame of transformed spectra
    """
    wavelengths = wavelengths if wavelengths is not None else df.index.astype(float)
    cr_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        spectrum = df[col].values
        try:
            points = np.column_stack((wavelengths, spectrum))
            hull = ConvexHull(points)

            # Get upper hull vertices sorted by wavelength
            hull_vertices = hull.vertices[np.argsort(wavelengths[hull.vertices])]
            upper = hull_vertices[np.diff(wavelengths[hull_vertices], prepend=0) >= 0]

            # Interpolate the continuum
            continuum = interp1d(wavelengths[upper], spectrum[upper],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')

            cr_spectrum = spectrum / continuum(wavelengths)
        except:
            cr_spectrum = np.ones_like(spectrum)  # fallback in degenerate cases

        if normalise:
            cr_min = np.min(cr_spectrum)
            cr_max = np.max(cr_spectrum)
            cr_spectrum = (cr_spectrum - cr_min) / (cr_max - cr_min + 1e-12)

        cr_df[col] = cr_spectrum

    return cr_df

# Example usage:
# df = spectra_wet.set_index("wavelength")
# wavelengths = df.index.astype(float)
# cr_df = continuum_removal(df, wavelengths=wavelengths, normalise=False) # continuum removal only
# crn_df = continuum_removal(df, wavelengths=wavelengths, normalise=True) # with band depth normalisation
# cr_df.insert(0, "wavelength", df.index) # Add wavelength column
# crn_df.insert(0, "wavelength", df.index) # Add wavelength column

def multiplicative_scatter_correction(df):
    """
    Apply Multiplicative Scatter Correction on spectra.

    Parameters:
    - df: pandas Dataframe of spectral data (rows = wavelengths, columns = samples)

    Returns:
    - DataFrame with same shape (MSC-corrected spectra)
    """
    df_corrected = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    
    # Mean spectrum (across all samples)
    mean_spectrum = df.mean(axis=1)

    for col in df.columns:
        spectrum = df[col]
        
        # Fit: x_i = a + b * mean_spectrum
        A = np.vstack([np.ones_like(mean_spectrum), mean_spectrum]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, spectrum.values, rcond=None)
        a, b = coeffs
        
        # Apply MSC
        corrected = (spectrum - a) / b
        df_corrected[col] = corrected

    return df_corrected

spectra = df.set_index("wavelength")

# Example usage:
# msc = multiplicative_scatter_correction(df)


def extended_multiplicative_scatter_correction(df, wavelengths):
    """
    Extended Multiplicative Scatter Correction (EMSC) using 2nd-order polynomial model.
    
    Parameters:
    - df: pandas DataFrame (rows = wavelengths, columns = samples)
    - wavelengths: array-like of same length as number of rows in df
    
    Returns:
    - DataFrame with EMSC-corrected spectra (same shape as input)
    """
    corrected_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    wavelengths = np.asarray(wavelengths)

    # Reference spectrum: mean of all samples
    reference = df.mean(axis=1).values

    # Construct design matrix with reference, lambda, lambda², and constant offset
    A = np.vstack([
        reference,
        wavelengths,
        wavelengths**2,
        np.ones_like(wavelengths)
    ]).T

    for col in df.columns:
        y = df[col].values
        # Solve least squares: z_i ≈ Abcd
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        b, d, e, a = coeffs

        # Reconstruct and subtract polynomial & offset, divide by multiplicative term
        baseline = d * wavelengths + e * wavelengths**2 + a
        corrected = (y - baseline) / b

        corrected_df[col] = corrected

    return corrected_df

# Example usage:
# df = spectra_dry.set_index("wavelength")
# wavelengths = df.index.astype(float)
# df_emsc = extended_multiplicative_scatter_correction(df, wavelengths)
# df_emsc.insert(0, "wavelength", df.index) # Add wavelength column
