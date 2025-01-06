import pandas as pd
import numpy as np
import os

def convolve_to_hyperion(lab_data, reflectance_prefix="spc.", srf_bandwidth=10, srf_file_path="hyperspectral/hyperion_wavelengths.csv"):
    """
    Convolves lab hyperspectral data to Hyperion bands using spectral response functions (SRFs).
    Automatically generates Hyperion SRFs from a provided wavelength file.
    Reflectance data MUST be in columns with prefix so that they can be recognised. E.g., reflectance_prefix="spc." for spc.400, spc.401...
    
    Parameters:
        lab_data (pd.DataFrame): Lab data with reflectance columns (e.g., starting with 'spc.<wavelength>').
        reflectance_prefix (str): Prefix for reflectance columns. Default is "spc." based on LUCAS database.
        srf_bandwidth (float): Assumed full-width half-maximum (FWHM) for each band (in nm). Default is 10.
        srf_file_path (str): Path to the Hyperion wavelength CSV file within GitHub.

    Returns:
        pd.DataFrame: Convolved data with retained metadata and Hyperion bands.
    """
    # Step 1: Load Hyperion Wavelengths
    if not os.path.exists(srf_file_path):
        raise FileNotFoundError(f"Hyperion wavelength file not found at {srf_file_path}")
    hyperion_wavelengths = pd.read_csv(srf_file_path)

    # Step 2: Create Hyperion SRFs
    hyperion_srf = {}
    for _, row in hyperion_wavelengths.iterrows():
        center = row['Wavelength']
        name = f"{row['Name']} - {center:.2f}"  # Include central wavelength in band name
        band_range = (center - srf_bandwidth / 2, center + srf_bandwidth / 2)
        wavelengths = np.linspace(band_range[0], band_range[1], 100)
        response = np.ones_like(wavelengths)  # Flat response
        hyperion_srf[name] = (wavelengths, response)

    # Step 3: Retain Metadata
    reflectance_columns = [col for col in lab_data.columns if col.startswith(reflectance_prefix)]
    metadata_columns = [col for col in lab_data.columns if col not in reflectance_columns]
    metadata = lab_data[metadata_columns]

    # Step 4: Extract Wavelengths and Reflectance Data
    wavelengths = np.array([float(col.replace(reflectance_prefix, "")) for col in reflectance_columns])
    reflectances = lab_data[reflectance_columns].values

    # Step 5: Perform Convolution
    convolved_data = {}
    for band, (band_wavelengths, band_responses) in hyperion_srf.items():
        interpolated_srf = np.interp(wavelengths, band_wavelengths, band_responses, left=0, right=0)
        band_values = np.sum(reflectances * interpolated_srf, axis=1) / np.sum(interpolated_srf)
        convolved_data[band] = band_values

    # Step 6: Combine Results
    convolved_df = pd.concat([metadata, pd.DataFrame(convolved_data)], axis=1)
    return convolved_df

# Example usage:
# lab_data = pd.read_csv('path_to_your_lab_data.csv')  # Your lab data
# convolved_data = convolve_to_hyperion(lab_data)
# convolved_data.to_csv('convolved_lab_data.csv', index=False)

def convolve_to_enmap(lab_data, reflectance_prefix="spc.", srf_file_path="hyperspectral/enmap_wavelengths.csv"):
    """
    Convolves lab hyperspectral data to EnMAP bands using spectral response functions (SRFs).
    Automatically generates EnMAP SRFs using FWHM from a provided wavelength file.
    Reflectance data MUST be in columns with prefix so that they can be recognised. E.g., reflectance_prefix="spc." for spc.400, spc.401...

    Parameters:
        lab_data (pd.DataFrame): Lab data with reflectance columns (starting with 'spc.<wavelength>').
        reflectance_prefix (str): Prefix for reflectance columns.
        srf_file_path (str): Path to the EnMAP wavelength CSV file.

    Returns:
        pd.DataFrame: Convolved data with retained metadata and EnMAP bands.
    """
    # Step 1: Load EnMAP Wavelengths
    if not os.path.exists(srf_file_path):
        raise FileNotFoundError(f"EnMAP wavelength file not found at {srf_file_path}")
    enmap_wavelengths = pd.read_csv(srf_file_path)

    # Step 2: Create EnMAP SRFs using FWHM
    enmap_srf = {}
    for _, row in enmap_wavelengths.iterrows():
        center = row['Wavelength']
        fwhm = row['FWHM']  # Retrieve FWHM for the band
        name = f"{row['Name']} - {center:.2f}"  # Include central wavelength in band name
        band_range = (center - fwhm / 2, center + fwhm / 2)  # Adjust range based on FWHM
        wavelengths = np.linspace(band_range[0], band_range[1], 100)
        response = np.ones_like(wavelengths)  # Flat response
        enmap_srf[name] = (wavelengths, response)

    # Step 3: Retain Metadata
    reflectance_columns = [col for col in lab_data.columns if col.startswith(reflectance_prefix)]
    metadata_columns = [col for col in lab_data.columns if col not in reflectance_columns]
    metadata = lab_data[metadata_columns]

    # Step 4: Extract Wavelengths and Reflectance Data
    wavelengths = np.array([float(col.replace(reflectance_prefix, "")) for col in reflectance_columns])
    reflectances = lab_data[reflectance_columns].values

    # Step 5: Perform Convolution
    convolved_data = {}
    for band, (band_wavelengths, band_responses) in enmap_srf.items():
        interpolated_srf = np.interp(wavelengths, band_wavelengths, band_responses, left=0, right=0)
        band_values = np.sum(reflectances * interpolated_srf, axis=1) / np.sum(interpolated_srf)
        convolved_data[band] = band_values

    # Step 6: Combine Results
    convolved_df = pd.concat([metadata, pd.DataFrame(convolved_data)], axis=1)
    return convolved_df