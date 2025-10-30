import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def load_data(stock_list):
    data = yf.download(stock_list, start="2025-09-01", interval="1d")
    prices = data['Close']
    rtn = np.log(prices / prices.shift(1)).dropna()
    return rtn
def plot_cor(cor_m):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cor_m, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
def plot_distribution(rtn,stock_list):
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
