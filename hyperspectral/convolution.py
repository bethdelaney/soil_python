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

def hyperion(spectra, Bands):
    """
    Hyperion convolution function.

    Args:
        spectra (pd.DataFrame): Hyperspectral reflectance data.
        Bands (pd.DataFrame): Spectral response functions (SRFs).
    """
    cwd = Path.cwd()
    bands_Dir = cwd / "bands"
    convolved_Dir = cwd / "convolved"

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

    # Convolve the spectra with the bands
    for column in spectra:
        f = Bands.mul(spectra[column], axis=0)
        band_file_path = bands_Dir / f'Bands_{column}.csv'
        f.to_csv(band_file_path)

    # Process each band file
    for band_file in bands_Dir.iterdir():
        file_name = band_file.stem
        convolution_process = pd.read_csv(band_file, index_col=0, header=0)

        # Convolution calculations for Hyperion bands
        convolved_values = []
        band_names = [
            "B008", "B009", "B010", "B011", "B012", "B013", "B014", "B015",
            "B016", "B017", "B018", "B019", "B020", "B021", "B022", "B023",
            "B024", "B025", "B026", "B027", "B028", "B029", "B030", "B031",
            "B032", "B033", "B034", "B035", "B036", "B037", "B038", "B039",
            "B040", "B041", "B042", "B043", "B044", "B045", "B046", "B047",
            "B048", "B049", "B050", "B051", "B052", "B053", "B054", "B055",
            "B056", "B057", "B077", "B078", "B079", "B080", "B081", "B082",
            "B083", "B084", "B085", "B086", "B087", "B088", "B089", "B090",
            "B091", "B092", "B093", "B094", "B095", "B096", "B097", "B098",
            "B099", "B100", "B101", "B102", "B103", "B104", "B105", "B106",
            "B107", "B108", "B109", "B110", "B111", "B112", "B113", "B114",
            "B115", "B116", "B117", "B118", "B119", "B120", "B121", "B122",
            "B123", "B124", "B125", "B126", "B127", "B128", "B129", "B130",
            "B131", "B132", "B133", "B134", "B135", "B136", "B137", "B138",
            "B139", "B140", "B141", "B142", "B143", "B144", "B145", "B146",
            "B147", "B148", "B149", "B150", "B151", "B152", "B153", "B154",
            "B155", "B156", "B157", "B158", "B159", "B160", "B161", "B162",
            "B163", "B164", "B165", "B166", "B167", "B168", "B169", "B170",
            "B171", "B172", "B173", "B174", "B175", "B176", "B177", "B178",
            "B179", "B180", "B181", "B182", "B183", "B184", "B185", "B186",
            "B187", "B188", "B189", "B190", "B191", "B192", "B193", "B194",
            "B195", "B196", "B197", "B198", "B199", "B200", "B201", "B202",
            "B203", "B204", "B205", "B206", "B207", "B208", "B209", "B210",
            "B211", "B212", "B213", "B214", "B215", "B216", "B217", "B218",
            "B219", "B220", "B221", "B222", "B223", "B224"
        ]

        central_wavelengths = [
            426.82, 436.99, 447.17, 457.34, 467.52, 477.69, 487.87, 498.04,
            508.22, 518.39, 528.57, 538.74, 548.92, 559.09, 569.27, 579.45,
            589.62, 599.8, 609.97, 620.15, 630.32, 640.5, 650.67, 660.85,
            671.02, 681.2, 691.37, 701.55, 711.72, 721.9, 732.07, 742.25,
            752.43, 762.6, 772.78, 782.95, 793.13, 803.3, 813.48, 823.65,
            833.83, 844.0, 854.18, 864.35, 874.53, 884.7, 894.88, 905.05,
            915.23, 925.41, 912.45, 922.54, 932.64, 942.73, 952.82, 962.91,
            972.99, 983.08, 993.17, 1003.3, 1013.3, 1023.4, 1033.49, 1043.59,
            1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19, 1114.19,
            1124.28, 1134.38, 1144.48, 1154.58, 1164.68, 1174.77, 1184.87,
            1194.97, 1205.07, 1215.17, 1225.17, 1235.27, 1245.36, 1255.46,
            1265.56, 1275.66, 1285.76, 1295.86, 1305.96, 1316.05, 1326.05,
            1336.15, 1346.25, 1356.35, 1366.45, 1376.55, 1386.65, 1396.74,
            1406.84, 1416.94, 1426.94, 1437.04, 1447.14, 1457.23, 1467.33,
            1477.43, 1487.53, 1497.63, 1507.73, 1517.83, 1527.92, 1537.92,
            1548.02, 1558.12, 1568.22, 1578.32, 1588.42, 1598.51, 1608.61,
            1618.71, 1628.81, 1638.81, 1648.9, 1659.0, 1669.1, 1679.2,
            1689.3, 1699.4, 1709.5, 1719.6, 1729.7, 1739.7, 1749.79, 1759.89,
            1769.99, 1780.09, 1790.19, 1800.29, 1810.38, 1820.48, 1830.58,
            1840.58, 1850.68, 1860.78, 1870.87, 1880.98, 1891.07, 1901.17,
            1911.27, 1921.37, 1931.47, 1941.57, 1951.57, 1961.66, 1971.76,
            1981.86, 1991.96, 2002.06, 2012.15, 2022.25, 2032.35, 2042.45,
            2052.45, 2062.55, 2072.65, 2082.75, 2092.84, 2102.94, 2113.04,
            2123.14, 2133.24, 2143.34, 2153.34, 2163.43, 2173.53, 2183.63,
            2193.73, 2203.83, 2213.93, 2224.03, 2234.12, 2244.22, 2254.22,
            2264.32, 2274.42, 2284.52, 2294.61, 2304.71, 2314.81, 2324.91,
            2335.01, 2345.11, 2355.21, 2365.2, 2375.3, 2385.4, 2395.5
        ]

        for band_name, center in zip(band_names, central_wavelengths):
            range_start = center - 5  # Define the range around central wavelength
            range_end = center + 5
            indices = (Bands.index >= range_start) & (Bands.index <= range_end)

            # Perform the convolution
            convolved_value = (
                np.trapz(convolution_process.loc[indices].squeeze(), dx=1) /
                np.trapz(Bands.loc[indices].squeeze(), dx=1)
            )
            convolved_values.append(convolved_value)

        # Save the convolved product
        convolved = {
            'Band name and centre wavelength (nm)': band_names,
            file_name + ' SRF': convolved_values
        }
        convolved_product = pd.DataFrame(convolved)
        convolved_product.set_index('Band name and centre wavelength (nm)', inplace=True)

        # Save to the convolved directory
        convolved_file_path = convolved_Dir / f'convolved_{file_name}.csv'
        convolved_product.to_csv(convolved_file_path)

        # Save to the convolved directory
        convolved_file_path = convolved_Dir / f'convolved_{file_name}.csv'
        convolved_product.to_csv(convolved_file_path)

    # Collate all convolved products
    collated_list = [pd.read_csv(convolved_file, index_col=0, header=0) for convolved_file in convolved_Dir.iterdir()]
    collated_convolved = pd.concat(collated_list, axis=1)

    # Save the collated results
    collated_convolved.to_csv(cwd / 'collated_convolved_bands.csv')

    # Optionally clear the directories (not delete)
    clear_directory(bands_Dir)
    clear_directory(convolved_Dir)

