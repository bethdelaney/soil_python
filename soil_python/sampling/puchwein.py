"""
Module for applying the Puchwein Algorithm to spectral data to derive sample calibration points.

Applies Principal Component Analysis (PCA) to reduce dimensionality.
Computes Mahalanobis distances for sample selection.
Uses Euclidean distances to ensure diversity in the selected sample set.
Includes functionality to display selected data and retrieve the coordinates of selected sample locations.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance

def mahalanobis_distance(x, mean, inv_cov):
    return distance.mahalanobis(x, mean, inv_cov)

def puchwein(X, pc=0.95, k=0.2, min_sel=5, details=False, center=True, scale=False):
    # Ensure X is a 2D numpy array
    X = np.array(X)
    
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 columns")
    if min_sel >= X.shape[0]:
        raise ValueError("min_sel must be lower than the number of rows in X")
    
    # PCA with centering and scaling
    pca = PCA(n_components=X.shape[1])
    X_transformed = pca.fit_transform(X)
    
    # Determine the number of principal components based on explained variance
    if pc < 1:
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= pc) + 1
        # Ensure we have at least 2 components
        num_components = max(num_components, 2)
    else:
        num_components = int(pc)
    
    X_reduced = X_transformed[:, :num_components]
    
    # Scale the principal components
    X_scaled = X_reduced / np.std(X_reduced, axis=0)
    
    # Compute Mahalanobis distance to the center
    center = np.mean(X_scaled, axis=0)
    cov_matrix = np.cov(X_scaled, rowvar=False)
    
    # Check the shape of the covariance matrix before computing the inverse
    print("Covariance matrix shape:", cov_matrix.shape)
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix is not square or 2D.")
    
    # Use the pseudoinverse to handle cases where the covariance matrix is singular
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    mahal_distances = np.array([mahalanobis_distance(sample, center, inv_cov_matrix) for sample in X_scaled])
    
    # Sort by Mahalanobis distance
    order = np.argsort(mahal_distances)[::-1]
    X_scaled = X_scaled[order]
    mahal_distances = mahal_distances[order]
    
    # Initial limiting distance
    d_ini = k * max((num_components - 2), 1)
    
    selected_indices = []
    all_selected_loops = []
    m = 1
    
    while True:
        Dm = m * d_ini
        selected = [0]  # Always start with the first sample (largest distance)
        
        for i in range(1, len(X_scaled)):
            dist_to_selected = np.min([distance.euclidean(X_scaled[i], X_scaled[j]) for j in selected])
            if dist_to_selected > Dm:
                selected.append(i)
        
        all_selected_loops.append(selected)
        
        if len(selected) <= min_sel:
            break
        m += 1
    
    # Compute leverage and optimise
    X_matrix = X_scaled
    leverage = np.diag(X_matrix @ np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T)
    observed_leverage = [np.sum(leverage[selected]) for selected in all_selected_loops]
    theoretical_leverage = (np.sum(leverage) / len(leverage)) * np.array([len(selected) for selected in all_selected_loops])
    leverage_diff = np.array(observed_leverage) - theoretical_leverage
    
    # Find the optimal loop with the maximum leverage difference
    optimal_loop = np.argmax(leverage_diff)
    model_indices = all_selected_loops[optimal_loop]
    
    if details:
        return {
            'model': order[model_indices],
            'test': np.setdiff1d(np.arange(len(X)), order[model_indices]),
            'pc': X_scaled,
            'loop_optimal': optimal_loop,
            'leverage': {
                'loop': np.arange(1, m),
                'observed_leverage': observed_leverage,
                'theoretical_leverage': theoretical_leverage,
                'leverage_diff': leverage_diff
            },
            'details': all_selected_loops
        }
    else:
        return {
            'model': order[model_indices],
            'test': np.setdiff1d(np.arange(len(X)), order[model_indices]),
            'pc': X_scaled,
            'loop_optimal': optimal_loop,
            'leverage': {
                'loop': np.arange(1, m),
                'observed_leverage': observed_leverage,
                'theoretical_leverage': theoretical_leverage,
                'leverage_diff': leverage_diff
            }
        }

def plot_pca(X, selected_pixels, n_components=2):
    """
    Perform PCA on the dataset and plot the first two principal components.
    
    Parameters:
    - X: 2D numpy array of the dataset (pixels x spectral bands)
    - selected_pixels: List or array of indices of selected samples
    - n_components: Number of PCA components to compute (default is 2)
    """
    
    # Run PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)  # PCA-transformed data
    
    # Reduce to the first two components for plotting
    X_reduced = X_transformed[:, :2]  # Use the first two components (PC1 and PC2)
    
    # Plot all samples
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='gray', label='All Samples')
    
    # Highlight the selected samples
    X_selected = X_reduced[selected_pixels]
    plt.scatter(X_selected[:, 0], X_selected[:, 1], color='red', label='Selected Samples')
    
    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Dataset with Selected Samples')
    
    # Add legend
    plt.legend()
    
    # Display grid
    plt.grid(True)
    
    # Show the plot
    plt.show()

def get_coordinates(selected_pixels, image_path):
    """
    Get the geographic coordinates (e.g., lat/lon) of the selected sample points.
    
    Parameters:
    - selected_pixels: Indices of selected samples from the Puchwein algorithm (1D array).
    - image_path: Path to the raster image from which the pixels were selected.
    
    Returns:
    - coordinates: List of tuples with the geographic coordinates (e.g., lat, lon) of each selected pixel.
    """
    # Open the raster using rasterio to get the geotransform (mapping from pixel to geographic coordinates)
    with rasterio.open(image_path) as src:
        # Get the raster's affine transformation matrix (maps pixel indices to geographic coordinates)
        transform = src.transform
        
        # Get the image shape
        rows, cols = src.height, src.width
        
        # Convert the 1D pixel indices to 2D row, col indices (assuming selected_pixels are flattened)
        row_col_indices = np.unravel_index(selected_pixels, (rows, cols))
        rows_selected = row_col_indices[0]
        cols_selected = row_col_indices[1]
        
        # Calculate geographic coordinates using the transform
        coordinates = []
        for row, col in zip(rows_selected, cols_selected):
            # Use the affine transformation to convert (row, col) to geographic (x, y) coordinates
            x, y = rasterio.transform.xy(transform, row, col)
            coordinates.append((x, y))  # Append the geographic coordinate (e.g., lon, lat or x, y)
    
    return coordinates
