# soil_python
This repository is dedicated to developing and implementing various soil remote sensing analysis packages, with a focus on utilising machine learning and satellite remote sensing data. The goal is to provide tools and algorithms that enhance soil property mapping estimation and other agricultural applications.

The repository will evolve over time, incorporating new methods and analysis techniques as they are developed or adapted. Current implementations are as follows:-

**Sampling:**
The Puchwein algorithm (Puchwein 1988) is implemented to select representative samples from a dataset based on Mahalanobis distance. This algorithm is particularly useful for soil spectral analysis, allowing the identification of the most dissimilar samples in a multivariate space, reducing redundancy in data, and optimising model performance. This algorithm can also be employed as a feature-based sampling strategy to determine the best locations for soil sampling on remote sensing spectral data. By analysing the spectral features from satellite imagery, the algorithm selects points that are most representative and dissimilar, ensuring that soil samples capture the full variability of the landscape.

**Hyperspectral:**
Convolution from laboratory hyperspectral data to various satellite sensor central wavelengths. Currently only Hyperion is applied.

Reference:
Puchwein, G. (1988). Selection of representative subsets by principal components. Communications in Soil Science and Plant Analysis, 19(7-12), 775-786. https://doi.org/10.1080/00103628809367971
