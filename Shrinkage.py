from optimization import cal_mean_var,mvp
import numpy as np

def shrinkage_mean_return(rtn,stock_list):
    """
    Calculate shrinkage estimator for mean returns using James-Stein type shrinkage.
    
    Shrinks sample mean returns toward the minimum variance portfolio (MVP) return,
    reducing estimation error in small samples.
    
    Parameters:
    -----------
    rtn : pandas.DataFrame
        Historical return data with shape (n_periods, n_assets)
    stock_list : list
        List of stock ticker symbols (length = n_assets)
    
    Returns:
    --------
    array
        Shrinkage estimator of mean returns (length n_assets)
        Formula: (1 - s_w) * sample_mean + s_w * MVP_mean
        where s_w is the shrinkage intensity parameter
    """
    n = rtn.shape[0]
    mean_rtn = rtn.mean().to_numpy()
    cov_m = rtn.cov().to_numpy()
    k = len(stock_list)
    w_mvp = mvp(mean_rtn,cov_m)
    mean_mvp = w_mvp.T @ mean_rtn
    inv_cov = ((n-k-2)/(n-1))*np.linalg.inv(cov_m)
    s_w = (k+2)/(k+2+n*(mean_rtn-mean_mvp).T @ inv_cov @ (mean_rtn-mean_mvp))
    shrinkage_mean = (1-s_w)*mean_rtn + s_w*mean_mvp
    return shrinkage_mean