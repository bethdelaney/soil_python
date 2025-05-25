# soil_python
This repository is dedicated to developing and implementing various soil remote sensing analysis packages, with a focus on utilising machine learning and spectral data. The goal is to provide tools and algorithms that enhance soil property mapping estimation and other agricultural applications.

The repository will evolve over time, incorporating new methods and analysis techniques as they are developed or adapted. Current implementations are as follows:-

## Sampling:

The Puchwein algorithm (Puchwein 1988) is implemented to select representative samples from a dataset based on Mahalanobis distance. This algorithm is particularly useful for soil spectral analysis, allowing the identification of the most dissimilar samples in a multivariate space, reducing redundancy in data, and optimising model performance. This algorithm can also be employed as a feature-based sampling strategy to determine the best locations for soil sampling on remote sensing spectral data. By analysing the spectral features from satellite imagery, the algorithm selects points that are most representative and dissimilar, ensuring that soil samples capture the full variability of the landscape.


## Hyperspectral:

Convolution from laboratory hyperspectral data to various satellite sensor central wavelengths. Currently Hyperion, EnMAP and Landsat Next are applied.

## Pretreatments:

A suite of spectral preprocessing techniques implemented to enhance spectral feature interpretation and reduce noise and scattering effects:
- SNV-DT: Standard Normal Variate followed by 2nd-order polynomial detrending to correct for baseline offsets and curvature (Buddenbaum & Steffens, 2012).
- MSC / EMSC: Multiplicative Scatter Correction and its extended form accounting for wavelength-dependent scatter effects (Buddenbaum & Steffens, 2012; Martens & Stark, 1991).
- CR / CRN: Continuum Removal and Normalised Continuum Removal (Band Depth Normalisation) to isolate and enhance absorption features across the full spectral range (Buddenbaum & Steffens, 2012).

## Prediction:

A flexible function to evaluate Support Vector Regression (SVR) on spectral data, with options for outlier removal (IQR), standardisation, PCA, and grid search hyperparameter tuning with cross-validation. The function returns R², RMSE, RPD, and the best SVR parameters, and is compatible with reflectance datasets formatted with wavelengths as columns and samples as rows.


## References:

- Puchwein, G., 1988. Selection of representative subsets by principal components. Communications in Soil Science and Plant Analysis, 19(7-12), 775-786. https://doi.org/10.1080/00103628809367971
- Buddenbaum, H. and Steffens, M., 2012. The effects of spectral pretreatments on chemometric analyses of soil profiles using laboratory imaging spectroscopy. Applied and Environmental Soil Science, 2012(1), p.274903. https://doi.org/10.1155/2012/274903
- Martens, H. and Stark, E., 1991. Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy. Journal of pharmaceutical and biomedical analysis, 9(8), pp.625-635. https://doi.org/10.1016/0731-7085(91)80188-F
