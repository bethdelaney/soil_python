"""Functions to convolve hyperspectral data to the equivalent bands of a selection of common satellite based sensors"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import shutil

def clear_directory(directory):
    """Remove all files and subdirectories in the directory"""
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

def S2(spectra, Bands):
    """Sentinel-2 convolution""" 
    cwd = Path.cwd()
    bands_Dir = cwd / "bands"
    convolved_Dir = cwd / "convolved"
    file_name = file_name

    # Create or clear the bands directory
    if bands_Dir.exists():
        clear_directory(bands_Dir)
    else:
        bands_Dir.mkdir()

    # Create or clear the convolved directory
    if convolved_Dir.exists():
        clear_directory(convolved_Dir)
    else:
        convolved_Dir.mkdir()

    # Check if water bands are removed
    while True:
        water_bands_removed = input("Are there regions where water bands have been removed in your data (Y or N)?: ")
        if water_bands_removed == "Y":
            np.seterr(divide="ignore")
            break
        elif water_bands_removed == "N":
            break
        else:
            print("Input a valid option (Y or N)")
            continue

    # Convolve the spectra with the bands
    for column in spectra:
        f = Bands.mul(spectra[column], axis=0)
        band_file_path = bands_Dir / f'Bands_{column}.csv'
        f.to_csv(band_file_path)

    # Process each band file
    for band_file in bands_Dir.iterdir():
        file_name = band_file.stem
        convolution_process = pd.read_csv(band_file, index_col=0, header=0)

        # Convolution calculations (updated with np.trapz)
        Band_2 = (np.trapz(convolution_process.iloc[89:184, 1], axis=0)) / (np.trapz(Bands.iloc[89:184, 1], axis=0))
        Band_3 = (np.trapz(convolution_process.iloc[188:233, 2], axis=0)) / (np.trapz(Bands.iloc[188:233, 2], axis=0))
        Band_4 = (np.trapz(convolution_process.iloc[296:344, 3], axis=0)) / (np.trapz(Bands.iloc[296:344, 3], axis=0))
        Band_5 = (np.trapz(convolution_process.iloc[345:364, 4], axis=0)) / (np.trapz(Bands.iloc[345:364, 4], axis=0))
        Band_6 = (np.trapz(convolution_process.iloc[381:399, 5], axis=0)) / (np.trapz(Bands.iloc[381:399, 5], axis=0))
        Band_7 = (np.trapz(convolution_process.iloc[419:447, 6], axis=0)) / (np.trapz(Bands.iloc[419:447, 6], axis=0))
        Band_8 = (np.trapz(convolution_process.iloc[423:557, 7], axis=0)) / (np.trapz(Bands.iloc[423:557, 7], axis=0))
        Band_8a = (np.trapz(convolution_process.iloc[497:531, 8], axis=0)) / (np.trapz(Bands.iloc[497:531, 8], axis=0))
        Band_11 = (np.trapz(convolution_process.iloc[1189:1332, 11], axis=0)) / (np.trapz(Bands.iloc[1189:1332, 11], axis=0))
        Band_12 = (np.trapz(convolution_process.iloc[1728:1970, 12], axis=0)) / (np.trapz(Bands.iloc[1728:1970, 12], axis=0))

        # Save the convolved product
        convolved = {
            'Band name and centre wavelength (nm)': ["Band 2 - 490", "Band 3 - 560", "Band 4 - 665", "Band 5 - 705",
                                                     "Band 6 - 740", "Band 7 - 783", "Band 8 - 842", "Band 8a - 865",
                                                     "Band 11 - 1610", "Band 12 - 2190"],
            file_name + ' SRF': [Band_2, Band_3, Band_4, Band_5, Band_6, Band_7, Band_8, Band_8a, Band_11, Band_12]
        }
        convolved_product = pd.DataFrame(convolved)
        convolved_product.set_index('Band name and centre wavelength (nm)', inplace=True)

        # Save to the convolved directory
        convolved_file_path = convolved_Dir / f'convolved_{file_name}.csv'
        convolved_product.to_csv(convolved_file_path)

    # Collate all convolved products
    collated_list = [pd.read_csv(convolved_file, index_col=0, header=0) for convolved_file in convolved_Dir.iterdir()]
    collated_convolved = pd.concat(collated_list, axis=1)

    # Save the collated results
    collated_convolved.to_csv(cwd / f'{file_name}_convolved_bands.csv')

    # Optionally clear the directories (not delete)
    clear_directory(bands_Dir)
    clear_directory(convolved_Dir)
