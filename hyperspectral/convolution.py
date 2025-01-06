"""Functions to convolve hyperspectral data to the equivalent bands of a selection of common satellite based sensors"""

import numpy as np
import pandas as pd

def create_hyperion_srf(hyperion_wavelengths, bandwidth=10):
    """
    Creates a spectral response function (SRF) for Hyperion bands.
    
    Parameters:
        hyperion_wavelengths (pd.DataFrame): DataFrame with 'Name' and 'Wavelength' columns.
        bandwidth (float): Assumed full-width half-maximum (FWHM) for each band (in nm).
        
    Returns:
        dict: Dictionary with band names as keys and (wavelength_range, response) tuples as values.
    """
    srf = {}
    for _, row in hyperion_wavelengths.iterrows():
        center = row['Wavelength']
        name = row['Name']
        # Simulate a Gaussian-like SRF (simple boxcar here for demonstration)
        band_range = (center - bandwidth / 2, center + bandwidth / 2)
        wavelengths = np.linspace(band_range[0], band_range[1], 100)
        response = np.ones_like(wavelengths)  # Flat response within band range
        srf[name] = (wavelengths, response)
    return srf

def convolve_to_hyperion(lab_data, hyperion_srf, reflectance_prefix="spc."):
    """
    Convolves lab hyperspectral data to Hyperion bands using spectral response functions (SRFs).

    Parameters:
        lab_data (pd.DataFrame): Lab data with reflectance columns (starting with 'spc.<wavelength>').
        hyperion_srf (dict): Dictionary with Hyperion band SRFs. Keys are band names,
                             values are tuples (wavelength_range, response).
        reflectance_prefix (str): Prefix for reflectance columns.

    Returns:
        pd.DataFrame: Convolved data with retained metadata and Hyperion bands.
    """
    # Retain metadata
    metadata_columns = ["Point_ID", "X", "Y", "OC"]
    metadata = lab_data[metadata_columns]

    # Extract wavelengths and reflectance data
    reflectance_columns = [col for col in lab_data.columns if col.startswith(reflectance_prefix)]
    wavelengths = np.array([float(col.replace(reflectance_prefix, "")) for col in reflectance_columns])
    reflectances = lab_data[reflectance_columns].values

    # Initialise dictionary for convolved data
    convolved_data = {}

    for band, (band_wavelengths, band_responses) in hyperion_srf.items():
        # Interpolate SRF to match hyperspectral wavelengths
        interpolated_srf = np.interp(wavelengths, band_wavelengths, band_responses, left=0, right=0)
        # Calculate weighted average for each sample
        band_values = np.sum(reflectances * interpolated_srf, axis=1) / np.sum(interpolated_srf)
        convolved_data[band] = band_values

    # Combine metadata and convolved bands
    convolved_df = pd.concat([metadata, pd.DataFrame(convolved_data)], axis=1)
    return convolved_df
    
