import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def GBM_simulation(vol_annual, mu_annual, S0, T, N, M, m_show=100):
    """
    Simulate Geometric Brownian Motion (GBM) paths for asset prices.

    Parameters:
    -----------
    vol_annual : float
        Annual volatility of the asset
    mu_annual : float
        Annual drift (expected return) of the asset
    S0 : float
        Initial stock price
    T : float
        Time horizon (in years)
    N : int
        Number of time steps
    M : int
        Number of simulation paths
    m_show : int, optional (default=100)
        Number of simulated paths to show in the plot

    Returns:
    --------
    None
        Plots simulated paths and terminal value histogram (with lognormal fit) as subplots,
        and displays mean and standard deviation of terminal values.
    """
    np.random.seed(8309)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    B = np.zeros((M, N + 1))
    for i in range(M):
        for j in range(1, N + 1):
            B[i, j] = B[i, j - 1] + np.random.normal(0, np.sqrt(dt))
    S = S0 * np.exp((mu_annual - 0.5 * vol_annual**2) * t[None, :] + vol_annual * B)

    value = S[:, -1]
    mean_value = np.mean(value)
    std_value = np.std(value)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot only m_show paths
    paths_to_show = min(m_show, M)
    for i in range(paths_to_show):
        axs[0].plot(t, S[i], alpha=0.8, lw=1)
    axs[0].set_xlabel('Time (years)')
    axs[0].set_ylabel('Price of Portfolio')
    axs[0].set_title(f'Simulated GBM Paths (showing {paths_to_show} of {M})')
    axs[0].grid(True, alpha=0.3)

    # Terminal value histogram and fitted lognormal PDF
    axs[1].hist(value, bins=40, density=True, alpha=0.6, color='skyblue', label="Simulated terminal values")
    shape, loc, scale = stats.lognorm.fit(value, floc=0)
    x = np.linspace(np.min(value), np.max(value), 300)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    axs[1].plot(x, pdf, 'r-', lw=2, label='Fitted Lognormal PDF')
    axs[1].set_xlabel("Terminal Portfolio Value")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Terminal Value Distribution w/ Lognormal Fit")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # Annotate mean and std on the histogram subplot
    annotation_text = f"Mean = {mean_value:.2f}\nStd = {std_value:.2f}"
    axs[1].text(
        0.98, 0.98, annotation_text,
        transform=axs[1].transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', alpha=0.15, color='gray')
    )

    plt.tight_layout()
    plt.show()
    return value