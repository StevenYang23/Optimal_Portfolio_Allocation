import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates

def load_data(stock_list):
    """
    Download stock data from Yahoo Finance and calculate log returns.
    
    Parameters:
    -----------
    stock_list : list
        List of stock ticker symbols to download
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing log returns for each stock with dates as index
    """
    data = yf.download(stock_list, start="2025-09-01", interval="1d")
    prices = data['Close']
    rtn = np.log(prices / prices.shift(1)).dropna()
    return rtn, prices

def plot_insight(cor_m, prices):
    """
    Plot correlation matrix as a heatmap and prices as line graphs in subplots,
    with the legend in the upper right to label each series. Ensures enough space for labels.

    Parameters:
    -----------
    cor_m : array-like
        Correlation matrix to visualize (2D array or DataFrame)
    prices : DataFrame
        DataFrame of asset prices (index: date, columns: tickers)

    Returns:
    --------
    None
        Displays a figure with two subplots: heatmap of correlations and price time series
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    # Heatmap subplot
    sns.heatmap(cor_m, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axs[0])
    axs[0].set_title('Correlation Matrix Heatmap')

    # Price line graph subplot with legend in upper right
    if hasattr(prices, "columns"):
        for col in prices.columns:
            axs[1].plot(prices.index, prices[col], label=str(col))
        axs[1].set_title('Price History')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        # Format date ticks if the index contains dates
        if prices.index.dtype.kind in {'M', 'm'}:
            axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        axs[1].legend(loc='upper right', fontsize=11, frameon=True)
    else:
        # fallback for numpy/array input
        axs[1].plot(prices)
        axs[1].set_title('Price History')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Price')

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # leave slightly more space on top
    plt.show()
    
def plot_distribution(rtn,stock_list):
    """
    Plot distribution of returns for each stock, comparing empirical distribution with Gaussian distribution.
    
    Parameters:
    -----------
    rtn : pandas.DataFrame
        DataFrame containing log returns for each stock
    stock_list : list
        List of stock ticker symbols to plot
    
    Returns:
    --------
    None
        Displays subplots showing return distributions for each stock with statistical information
    """
    warnings.filterwarnings('ignore')
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    # Calculate the optimal grid size
    n_stocks = len(stock_list)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols  # Ceiling division
    # Create subplots with dynamic grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration
    # Plot individual stock distributions (log returns approach)
    for i, stock in enumerate(stock_list):
        if i < len(axes):
            returns = rtn[stock].dropna()
            # Calculate statistics
            mu = returns.mean() * 100  # Convert to percentage for display
            std = returns.std() * 100   # Convert to percentage for display
            n_observations = len(returns)
            # Plot histogram with KDE
            sns.histplot(returns * 100, stat='density', alpha=0.6, 
                        label='Empirical Distribution', ax=axes[i])
            # Generate Gaussian distribution using calculated mean and std
            x = np.linspace(returns.min() * 100, returns.max() * 100, 1000)
            gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
            # Plot Gaussian distribution
            axes[i].plot(x, gaussian, 'r-', linewidth=2, label='Gaussian Distribution')
            # Customize plot
            axes[i].set_title(f'{stock}\n'
                            f'μ={mu:.3f}%, σ={std:.3f}%',
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Monthly Log Return (%)')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            # Add text box with statistics
            textstr = f'N = {n_observations}\nMin = {returns.min()*100:.2f}%\nMax = {returns.max()*100:.2f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
    # Remove any empty subplots
    for i in range(len(stock_list), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.suptitle('Monthly Log Returns Distribution with Gaussian Distribution (2024 Training Data)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.show()
