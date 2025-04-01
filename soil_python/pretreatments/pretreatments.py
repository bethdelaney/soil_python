import numpy as np
import pandas as pd
from scipy.signal import detrend
from numpy.polynomial import Polynomial

def snv_detrend(X):
    """
    Apply Standard Normal Variate (SNV) and 2nd order polynomial detrending to spectral data.

    Parameters:
    - X: 2D numpy array or pandas DataFrame (rows = samples, columns = wavelengths)

    Returns:
    - SNV-DT transformed spectra as numpy array
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    snv_data = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    
    detrended_data = np.zeros_like(snv_data)

    wavelengths = np.arange(snv_data.shape[1])

    for i in range(snv_data.shape[0]):
        coeffs = Polynomial.fit(wavelengths, snv_data[i, :], deg=2).convert().coef
        baseline = coeffs[0] + coeffs[1]*wavelengths + coeffs[2]*wavelengths**2
        detrended_data[i, :] = snv_data[i, :] - baseline

    return detrended_data

df = spectra_wet

# Apply SNV and detrending
snv_detrended_spectra = snv_detrend(df.iloc[:, 1:].values)

# Convert back to DataFrame
snv_detrended_df = pd.DataFrame(snv_detrended_spectra, columns=df.columns[1:])

# Write csv
snv_detrended_df.insert(0, "wavelength", df["wavelength"].values)
snv_detrended_df.to_csv(f'C:\\Users\\bd167\\OneDrive - University of Leicester\\Documents\\Data\\Spec_Data\\Data\\CSVs\\results\\wet_snv_detrended.csv', index=False)
