import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def cal_mean_var(weights, mean_np, cov_np): 
    portfolio_return = weights.T @ mean_np
    portfolio_variance = weights.T @ (cov_np @ weights)
    return portfolio_return, portfolio_variance
def cal_util(mu,var,A):
    return mu - 0.5*A*var
def RtnPerRisk(mean_rtn,cov_m):
    w = np.linalg.inv(cov_m) @ mean_rtn
    w = w/sum(w)
    return w
def mvp(mean_rtn,cov_m,target_return=None):
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
    ones = np.ones(len(mean_rtn))
    Sigma_inv = np.linalg.inv(cov_m)
    w = Sigma_inv @ (mean_rtn - r*ones)
    w = w/sum(w)
    return w
def Max_util(mean_rtn,cov_m,A):
    ones = np.ones(len(mean_rtn))
    Sigma_inv = np.linalg.inv(cov_m)
    lam = (ones @ Sigma_inv @ mean_rtn - A) / (ones @ Sigma_inv @ ones)
    w = (1/A) * Sigma_inv @ (mean_rtn - lam * ones)
    w = w/sum(w)
    return w
def plot_port(w, mean_rtn, cov_m,rtn,stock_list,A):
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
def expand_weights(w_active, short_res):
    w_full = np.zeros(len(short_res))
    idx = 0
    for i, flag in enumerate(short_res):
        if flag == 1:
            w_full[i] = w_active[idx]
            idx += 1
    return w_full
def active(short_res,mean_rtn,cov_m):
    active_idx = [i for i, s in enumerate(short_res) if s == 1]
    return mean_rtn[active_idx],cov_m[np.ix_(active_idx, active_idx)]