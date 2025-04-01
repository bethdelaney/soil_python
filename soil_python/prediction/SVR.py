import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

def evaluate_svr_model(df, target_col, wavelength_start='350',
                       remove_outliers=False, standardise=False, use_pca=False,
                       pca_components=0.99, cv_splits=5, verbose=True):
    """
    Evaluates SVR model with optional standardisation, PCA, and outlier removal.

    Parameters:
    - df: pandas DataFrame
        Input DataFrame containing spectral data and the target variable. 
        Must have wavelengths as columns (as strings or numbers) and one column as the target.

    - target_col: str
        Name of the column containing the target variable (e.g., 'OC').

    - wavelength_start: str or float, default='350'
        First wavelength column to include as predictor (assumes all later columns are wavelengths).

    - remove_outliers: bool, default=False
        If True, removes outliers from the target column using IQR method before training.

    - standardise: bool, default=False
        If True, applies `StandardScaler()` to the input features.

    - use_pca: bool, default=False
        If True, applies PCA to reduce spectral dimensionality before SVR.

    - pca_components: float or int, default=0.99
        Number of PCA components to retain. If float, selects enough components to explain this proportion of variance.

    - cv_splits: int, default=5
        Number of folds in K-Fold cross-validation.

    - verbose: bool, default=True
        If True, prints the R², RMSE, RPD, and best hyperparameters to console.

    Returns:
    Dictionary containing:
          - 'R2': Mean R² score
          - 'RMSE': Mean root mean squared error
          - 'RPD': Ratio of standard deviation to RMSE
          - 'Best Params': Best SVR hyperparameters from grid search
    """
    df = df.copy()
    if remove_outliers:
        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]

    # Extract features and target
    X = df.loc[:, wavelength_start:].astype(float).values
    y = df[target_col].astype(float).values

    # Pipeline steps
    steps = []
    if standardise:
        steps.append(('scaler', StandardScaler()))
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_components)))
    steps.append(('svr', SVR()))

    pipeline = Pipeline(steps)

    # Hyperparameter grid
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__gamma': ['scale']
    }

    # Cross-validation config
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Best model and evaluation
    best_model = grid_search.best_estimator_

    scoring = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error'
    }

    cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring)
    mean_r2 = np.mean(cv_results['test_r2'])
    mean_rmse = -np.mean(cv_results['test_rmse'])
    std_y = np.std(y)
    rpd = std_y / mean_rmse

    if verbose:
        print(f"R²: {mean_r2:.3f}")
        print(f"RMSE: {mean_rmse:.3f}")
        print(f"RPD: {rpd:.3f}")
        print(f"Best Params: {grid_search.best_params_}")

    return {
        'R2': round(mean_r2, 3),
        'RMSE': round(mean_rmse, 3),
        'RPD': round(rpd, 3),
        'Best Params': grid_search.best_params_
    }

# Example usage:
# result = evaluate_svr_model(df, target_col="OC", remove_outliers=False, standardise=False, use_pca=False)
