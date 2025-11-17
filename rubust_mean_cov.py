import numpy as np

def campbell_robust_est(rtn, b1=2.0, b2=1.25):
    """
    Estimate robust mean and covariance matrix using Campbell's robust estimation method.
    
    This method uses Mahalanobis distances to identify and downweight outliers,
    providing more robust estimates than standard sample mean and covariance.
    
    Parameters:
    -----------
    rtn : array-like or pandas.DataFrame
        Return data with shape (n_samples, n_assets)
    b1 : float, default=2.0
        Threshold parameter for Mahalanobis distance (d0 = sqrt(k) + b1/sqrt(2))
    b2 : float, default=1.25
        Bandwidth parameter for weight function when d > d0
    
    Returns:
    --------
    tuple
        (robust_mean, robust_cov) - Robust estimates of mean returns and covariance matrix
        - robust_mean: array of shape (n_assets,)
        - robust_cov: array of shape (n_assets, n_assets)
    
    Notes:
    ------
    Observations with Mahalanobis distance <= d0 get weight 1.0.
    Observations beyond d0 get exponentially decreasing weights based on b2.
    """
    # Ensure rtn is a DataFrame or 2D array
    if hasattr(rtn, 'values'):
        r = rtn.values  # shape (n, k)
    else:
        r = np.asarray(rtn)
    
    n, k = r.shape
    
    # Step 1: Initial estimates
    mean_rtn = r.mean(axis=0)               # shape (k,)
    cov_rtn = np.cov(r, rowvar=False, ddof=1)  # shape (k, k)
    
    # Step 2: Inverse covariance (once)
    try:
        inv_cov = np.linalg.inv(cov_rtn)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_rtn)  # fallback
    
    # Step 3: d0 = sqrt(k) + b1 / sqrt(2)
    d0 = np.sqrt(k) + b1 / np.sqrt(2)
    
    # Step 4: Compute Mahalanobis distances and weights
    d = np.empty(n)
    w = np.empty(n)
    
    for i in range(n):
        ex = r[i] - mean_rtn               # shape (k,)
        d[i] = np.sqrt(ex @ inv_cov @ ex)  # scalar
        if d[i] <= d0:
            w[i] = 1.0
        else:
            A_di = d0 * np.exp(- (d[i] - d0)**2 / (2 * b2**2))
            w[i] = A_di / d[i]
    
    # Step 5: Robust mean
    sum_w = w.sum()
    new_mean_rtn = (w[:, None] * r).sum(axis=0) / sum_w  # shape (k,)
    
    # Step 6: Robust covariance (Campbell formula: weighted, centered on new mean)
    centered = r - new_mean_rtn            # shape (n, k)
    w2 = w ** 2
    numerator = np.einsum('i,ij,ik->jk', w2, centered, centered)  # ∑ w_i² (r_i - μ)(r_i - μ)^T
    denominator = w2.sum() - 1
    if denominator <= 0:
        raise ValueError("Denominator ≤ 0. Check weights or sample size.")
    new_cov_rtn = numerator / denominator
    
    return new_mean_rtn, new_cov_rtn