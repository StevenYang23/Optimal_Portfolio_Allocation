import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def cal_mean_var(weights, mean_np, cov_np): 
    """
    Calculate portfolio expected return and variance given weights, mean returns, and covariance matrix.
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights (vector of length n)
    mean_np : array-like
        Expected returns vector (length n)
    cov_np : array-like
        Covariance matrix (n x n)
    
    Returns:
    --------
    tuple
        (portfolio_return, portfolio_variance) - Expected return and variance of the portfolio
    """
    portfolio_return = weights.T @ mean_np
    portfolio_variance = weights.T @ (cov_np @ weights)
    return portfolio_return, portfolio_variance

def cal_util(mu,var,A):
    """
    Calculate utility using mean-variance utility function.
    
    Parameters:
    -----------
    mu : float
        Expected return
    var : float
        Variance (risk measure)
    A : float
        Risk aversion parameter (higher A = more risk averse)
    
    Returns:
    --------
    float
        Utility value: U = μ - 0.5 * A * σ²
    """
    return mu - 0.5*A*var

def RtnPerRisk(mean_rtn,cov_m):
    """
    Calculate weights that maximize return per unit of risk (inverse variance weighting).
    
    Parameters:
    -----------
    mean_rtn : array-like
        Expected returns vector (length n)
    cov_m : array-like
        Covariance matrix (n x n)
    
    Returns:
    --------
    array
        Normalized portfolio weights (sum to 1)
    """
    w = np.linalg.inv(cov_m) @ mean_rtn
    w = w/sum(w)
    return w

def mvp(mean_rtn,cov_m,target_return=None):
    """
    Calculate Minimum Variance Portfolio (MVP) weights.
    
    If target_return is None, returns the global minimum variance portfolio.
    Otherwise, returns the minimum variance portfolio with the specified target return.
    
    Parameters:
    -----------
    mean_rtn : array-like
        Expected returns vector (length n)
    cov_m : array-like
        Covariance matrix (n x n)
    target_return : float, optional
        Target portfolio return. If None, calculates global minimum variance portfolio.
    
    Returns:
    --------
    array
        Normalized portfolio weights (sum to 1)
    """
    ones = np.ones(len(mean_rtn))
    Sigma_inv = np.linalg.inv(cov_m)
    if target_return is None:
        w = Sigma_inv @ ones
    else:
        A = ones @ Sigma_inv @ ones
        B = ones @ Sigma_inv @ mean_rtn
        C = mean_rtn @ Sigma_inv @ mean_rtn
        D = A * C - B**2
        lambda_ = (A * target_return - B) / D
        gamma_   = (C - B * target_return) / D
        w = lambda_ * (Sigma_inv @ mean_rtn) + gamma_ * (Sigma_inv @ ones)
    w = w/sum(w)
    return w

def MaxSharpe(mean_rtn,cov_m,r):
    """
    Calculate portfolio weights that maximize Sharpe ratio.
    
    Parameters:
    -----------
    mean_rtn : array-like
        Expected returns vector (length n)
    cov_m : array-like
        Covariance matrix (n x n)
    r : float
        Risk-free rate
    
    Returns:
    --------
    array
        Normalized portfolio weights (sum to 1) that maximize (μ_p - r) / σ_p
    """
    ones = np.ones(len(mean_rtn))
    Sigma_inv = np.linalg.inv(cov_m)
    w = Sigma_inv @ (mean_rtn - r*ones)
    w = w/sum(w)
    return w

def Max_util(mean_rtn,cov_m,A):
    """
    Calculate portfolio weights that maximize utility function U = μ - 0.5*A*σ².
    
    Parameters:
    -----------
    mean_rtn : array-like
        Expected returns vector (length n)
    cov_m : array-like
        Covariance matrix (n x n)
    A : float
        Risk aversion parameter (higher A = more risk averse)
    
    Returns:
    --------
    array
        Normalized portfolio weights (sum to 1) that maximize utility
    """
    ones = np.ones(len(mean_rtn))
    Sigma_inv = np.linalg.inv(cov_m)
    lam = (ones @ Sigma_inv @ mean_rtn - A) / (ones @ Sigma_inv @ ones)
    w = (1/A) * Sigma_inv @ (mean_rtn - lam * ones)
    w = w/sum(w)
    return w

def plot_port(w, mean_rtn, cov_m,rtn,stock_list,A):
    """
    Plot portfolio analysis including cumulative returns, return distribution, and weight allocation.
    
    Creates three subplots:
    1. Cumulative return over time
    2. Histogram of returns with normal distribution overlay and statistics
    3. Horizontal bar chart showing portfolio weights by stock
    
    Parameters:
    -----------
    w : array-like
        Portfolio weights (length n)
    mean_rtn : array-like
        Expected returns vector (length n)
    cov_m : array-like
        Covariance matrix (n x n)
    rtn : pandas.DataFrame
        Historical returns for backtesting
    stock_list : list
        List of stock ticker symbols
    A : float
        Risk aversion parameter for utility calculation
    
    Returns:
    --------
    tuple
        (portfolio_return, portfolio_variance) - Expected return and variance of the portfolio
    """
    portfolio_return, portfolio_variance = cal_mean_var(w, mean_rtn, cov_m)
    utility = cal_util(portfolio_return, portfolio_variance, A)
    back_test_rtn = rtn@w
    cum_return = [1]
    for i in range(len(back_test_rtn)):
        cum_return.append(cum_return[-1]*(1+back_test_rtn[i]))
    cum_return = np.array(cum_return) - 1
    # Display portfolio return and variance on the histogram subplot, and overlay normal distribution
    mu = portfolio_return * 100  # Convert to %
    sigma = np.sqrt(portfolio_variance) * 100  # Convert to %
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(cum_return*100, label='Cumulative Return(%)', color='blue')
    axes[0].set_title('Cumulative Return Over Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Cumulative Return (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f'Cumulative Return Over Time\n(Utility = {utility:.4f})')
    hist_vals, bins, patches = axes[1].hist(back_test_rtn * 100, bins=30, density=True, alpha=0.6, color='skyblue', label='Returns Histogram')
    axes[1].axvline(mu, color='red', linestyle='--', label='Mean (%.2f%%)' % mu)
    # Display mean and variance beside "Utility" in the title
    axes[1].set_title(f"Cumulative Return Over Time\nUtility = {utility:.4f} | Mean = {mu:.2f}% | Var = {portfolio_variance*10000:.2f}")
    x = np.linspace((back_test_rtn * 100).min(), (back_test_rtn * 100).max(), 200)
    pdf = norm.pdf(x, mu, sigma)
    axes[1].plot(x, pdf * (hist_vals.max()/pdf.max()), color='red', linestyle='-', linewidth=1.5, label='Normal Dist')
    axes[1].legend()

    colors = ['green' if weight > 0 else 'red' for weight in w]
    bars = axes[2].barh(stock_list, w, color=colors, alpha=0.7)
    axes[2].set_xlabel('Weight Allocation')
    axes[2].set_title('Portfolio Allocation')
    axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    # Annotate the value of each weight under the stock name (tick label)
    axes[2].set_yticks(range(len(stock_list)))
    axes[2].set_yticklabels([
        f"{stock}\n{weight:.2%}" for stock, weight in zip(stock_list, w)
    ])
    plt.tight_layout()
    plt.show()
    return portfolio_return, portfolio_variance

def expand_weights(w_active, short_res):
    """
    Expand active portfolio weights to full universe based on short selling constraints.
    
    Parameters:
    -----------
    w_active : array-like
        Portfolio weights for active assets only (length = number of active assets)
    short_res : array-like
        Binary vector indicating which assets are active (1 = active, 0 = inactive)
        Length equals total number of assets
    
    Returns:
    --------
    array
        Full weight vector (length = total assets) with zeros for inactive assets
    """
    w_full = np.zeros(len(short_res))
    idx = 0
    for i, flag in enumerate(short_res):
        if flag == 1:
            w_full[i] = w_active[idx]
            idx += 1
    return w_full
    
def active(short_res,mean_rtn,cov_m):
    """
    Extract mean returns and covariance matrix for active assets only.
    
    Parameters:
    -----------
    short_res : array-like
        Binary vector indicating which assets are active (1 = active, 0 = inactive)
    mean_rtn : array-like
        Full expected returns vector (length = total assets)
    cov_m : array-like
        Full covariance matrix (n x n where n = total assets)
    
    Returns:
    --------
    tuple
        (mean_rtn_active, cov_m_active) - Mean returns and covariance matrix for active assets only
    """
    active_idx = [i for i, s in enumerate(short_res) if s == 1]
    return mean_rtn[active_idx],cov_m[np.ix_(active_idx, active_idx)]