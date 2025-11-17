# 📊 Optimal Portfolio Allocation

A comprehensive Python project for optimal portfolio allocation using various optimization strategies, robust estimation methods, and Monte Carlo simulation. This project implements modern portfolio theory techniques to construct efficient portfolios and analyze their performance.

## Features

- **Data Processing**: Download and process stock data from Yahoo Finance
- **Robust Estimation**: Campbell's robust method for mean and covariance estimation to handle outliers
- **Shrinkage Estimators**: James-Stein type shrinkage for reducing estimation error
- **Portfolio Optimization**: Multiple optimization strategies including:
  - Maximum Utility Portfolio
  - Minimum Variance Portfolio (MVP)
  - Maximum Sharpe Ratio Portfolio
  - Return per Risk Portfolio
- **Visualization**: Comprehensive plots for data analysis, portfolio performance, and distributions
- **Monte Carlo Simulation**: Geometric Brownian Motion simulation for portfolio value forecasting

## Visualization Examples

### Data Insights
![Data Insight](demo/Data_insight.png)

The data insight plot shows:
- Correlation matrix heatmap displaying relationships between assets
- Price history time series for all stocks in the portfolio

### Return Distribution Analysis
![Distribution List](demo/dist_list.png)

This visualization compares empirical return distributions with Gaussian distributions for each stock, providing insights into the distribution characteristics and normality assumptions.

### Portfolio Performance and Weights
![BackTest and Report Weight](demo/BackTest_and_report_weight.png)

Comprehensive portfolio analysis including:
- Cumulative return over time
- Return distribution histogram with normal distribution overlay
- Portfolio weight allocation across assets

### Monte Carlo Simulation
![MC Simulation](demo/MC_sim.png)

Geometric Brownian Motion simulation results:
- Simulated price paths over time
- Terminal value distribution with lognormal fit
- Statistical summary (mean and standard deviation)

## Portfolio Optimization Strategies

This project implements several advanced portfolio optimization strategies based on Modern Portfolio Theory (MPT). Each strategy solves a different optimization problem to find optimal asset weights that satisfy different objectives and constraints.

### 1. Maximum Utility Portfolio (`Max_util`)

**Objective**: Maximize investor utility based on mean-variance preference

**Mathematical Formulation**:
- Maximize: U(w) = w^T μ - 0.5 × A × w^T Σ w
- Subject to: w^T 1 = 1 (weights sum to 1)

Where:
- `w`: Portfolio weight vector
- `μ`: Expected returns vector
- `Σ`: Covariance matrix
- `A`: Risk aversion parameter

**Use Cases**:
- Customizable risk-return tradeoff based on investor preferences
- Direct incorporation of investor's risk tolerance
- Suitable for investors with clear utility preferences

**Key Features**:
- Higher `A` values lead to more conservative portfolios (higher weight on low-risk assets)
- Lower `A` values allow for more aggressive strategies
- Closed-form analytical solution available

**Implementation**:
```python
A = 200  # Risk aversion parameter (higher = more risk averse)
w = Max_util(mean_rtn, cov_m, A)
```

**Advantages**:
- Explicitly models investor risk preferences
- Provides intuitive control over risk-return tradeoff
- Well-established theoretical foundation

---

### 2. Minimum Variance Portfolio (`mvp`)

**Objective**: Minimize portfolio variance/volatility

**Mathematical Formulation**:
- Minimize: w^T Σ w
- Subject to: w^T 1 = 1

**Constrained Version** (with target return):
- Minimize: w^T Σ w
- Subject to: w^T 1 = 1 and w^T μ = μ_target

Where:
- `μ_target`: Target expected return

**Use Cases**:
- Risk-averse investors prioritizing volatility reduction
- Foundation portfolio for other optimization strategies
- Long-only or long-short implementations

**Key Features**:
- Provides the minimum risk portfolio on the efficient frontier
- Can be constrained to achieve a specific target return
- Analytical solution: w* = (Σ^(-1) × 1) / (1^T × Σ^(-1) × 1)

**Implementation**:
```python
# Global minimum variance portfolio
w = mvp(mean_rtn, cov_m)

# Minimum variance with target return
target_return = 0.005
w = mvp(mean_rtn, cov_m, target_return=target_return)
```

**Advantages**:
- Minimizes downside risk
- Well-suited for risk-averse investors
- Provides efficient frontier endpoint

---

### 3. Maximum Sharpe Ratio Portfolio (`MaxSharpe`)

**Objective**: Maximize risk-adjusted return (Sharpe Ratio)

**Mathematical Formulation**:
- Maximize: SR = (w^T μ - r) / √(w^T Σ w)
- Subject to: w^T 1 = 1

Where:
- `r`: Risk-free rate
- `SR`: Sharpe Ratio (excess return per unit of risk)

**Use Cases**:
- Optimal portfolio when risk-free asset is available
- Maximizing risk-adjusted returns
- Benchmarking portfolio performance

**Key Features**:
- Finds the tangency portfolio on the efficient frontier
- Maximizes return per unit of risk
- Analytical solution: w* ∝ Σ^(-1) × (μ - r × 1)

**Implementation**:
```python
r = 0.045  # Annual risk-free rate
r_daily = r / 252  # Convert to daily rate
w = MaxSharpe(mean_rtn, cov_m, r_daily)
```

**Advantages**:
- Optimal portfolio in mean-variance space when risk-free asset exists
- Best risk-adjusted performance metric
- Industry standard for portfolio evaluation

---

### 4. Return per Risk Portfolio (`RtnPerRisk`)

**Objective**: Maximize return per unit of risk using inverse variance weighting

**Mathematical Formulation**:
- Weights: w ∝ Σ^(-1) × μ
- Normalized: w = (Σ^(-1) × μ) / (1^T × Σ^(-1) × μ)

**Use Cases**:
- Simple risk-adjusted weighting scheme
- Inverse volatility weighting with return adjustments
- Quick portfolio construction

**Key Features**:
- Inverse covariance weighting adjusted for expected returns
- Simpler alternative to full optimization
- Computationally efficient

**Implementation**:
```python
w = RtnPerRisk(mean_rtn, cov_m)
```

**Advantages**:
- Fast computation
- Intuitive weighting scheme
- Good baseline for comparison

## Robust Estimation Methods

Accurate estimation of mean returns and covariance matrices is critical for portfolio optimization. However, financial data often contains outliers and estimation errors. This project implements advanced robust estimation techniques to handle these challenges.

### Campbell's Robust Estimation (`campbell_robust_est`)

**Problem Addressed**: Standard sample mean and covariance estimators are sensitive to outliers, which are common in financial returns data. A few extreme returns can significantly distort portfolio weights.

**Methodology**: Campbell's robust estimation method uses Mahalanobis distances to identify and downweight outliers, providing more reliable estimates of mean returns and covariance.

**Mathematical Framework**:

1. **Initial Estimates**:
   - Sample mean: μ̂₀ = (1/n) Σᵢ rᵢ
   - Sample covariance: Σ̂₀ = (1/(n-1)) Σᵢ (rᵢ - μ̂₀)(rᵢ - μ̂₀)^T

2. **Mahalanobis Distance Calculation**:
   - For each observation i: dᵢ = √[(rᵢ - μ̂₀)^T × Σ̂₀^(-1) × (rᵢ - μ̂₀)]
   - Threshold: d₀ = √k + b₁/√2
     - `k`: Number of assets
     - `b₁`: Threshold parameter (default: 2.0)

3. **Weight Assignment**:
   - If dᵢ ≤ d₀: wᵢ = 1.0 (full weight)
   - If dᵢ > d₀: wᵢ = (d₀ × exp[-(dᵢ - d₀)²/(2b₂²)]) / dᵢ
     - `b₂`: Bandwidth parameter (default: 1.25)

4. **Robust Mean**:
   - μ̂_robust = Σᵢ wᵢ rᵢ / Σᵢ wᵢ

5. **Robust Covariance**:
   - Σ̂_robust = [Σᵢ wᵢ²(rᵢ - μ̂_robust)(rᵢ - μ̂_robust)^T] / [Σᵢ wᵢ² - 1]

**Key Parameters**:
- `b1` (default: 2.0): Controls the threshold distance beyond which observations are downweighted
  - Higher values allow more extreme observations before downweighting
  - Lower values are more aggressive in downweighting outliers
- `b2` (default: 1.25): Controls the rate of weight decay for outliers
  - Higher values provide smoother weight transitions
  - Lower values cause steeper weight reductions

**Use Cases**:
- Portfolio optimization with noisy or outlier-contaminated data
- Handling market crashes or extreme events in return data
- Improving portfolio stability in volatile markets
- Reducing the impact of data errors or reporting anomalies

**Implementation**:
```python
mean_rtn_robust, cov_m_robust = campbell_robust_est(rtn, b1=2.0, b2=1.25)
```

**Advantages**:
- Reduces impact of outliers on portfolio weights
- More stable portfolio allocations
- Better performance in presence of fat-tailed return distributions
- Automatic identification and downweighting of extreme observations

**When to Use**:
- Working with high-volatility assets
- Data contains suspected outliers or errors
- Portfolio weights seem unstable or extreme
- Market conditions include unusual events

---

### Shrinkage Estimator (`shrinkage_mean_return`)

**Problem Addressed**: In small samples or when the number of assets approaches the sample size, sample mean returns have high estimation error. The James-Stein paradox shows that shrinking sample means toward a common target can reduce total mean-squared error.

**Methodology**: The shrinkage estimator shrinks sample mean returns toward the minimum variance portfolio (MVP) return, effectively borrowing strength across assets to reduce estimation error.

**Mathematical Framework**:

1. **Sample Estimates**:
   - Sample mean: μ̂ = (1/n) Σᵢ rᵢ
   - Sample covariance: Σ̂ = (1/(n-1)) Σᵢ (rᵢ - μ̂)(rᵢ - μ̂)^T

2. **MVP Calculation**:
   - MVP weights: w_MVP = (Σ̂^(-1) × 1) / (1^T × Σ̂^(-1) × 1)
   - MVP return: μ_MVP = w_MVP^T × μ̂

3. **Shrinkage Intensity**:
   - s_w = (k + 2) / (k + 2 + n × (μ̂ - μ_MVP)^T × [(n-k-2)/(n-1)] × Σ̂^(-1) × (μ̂ - μ_MVP))
   - Where:
     - `k`: Number of assets
     - `n`: Number of observations
     - Denominator is adjusted inverse covariance matrix

4. **Shrunk Mean**:
   - μ̂_shrink = (1 - s_w) × μ̂ + s_w × μ_MVP × 1
   - Where `1` is a vector of ones

**Interpretation**:
- When sample means are similar to MVP return → s_w → 1 (more shrinkage)
- When sample means differ significantly → s_w → 0 (less shrinkage)
- Shrinkage intensity automatically adjusts based on data quality

**Key Properties**:
- **Adaptive Shrinkage**: Shrinkage intensity depends on the data
- **Theoretical Optimality**: Minimizes expected mean-squared error
- **Small Sample Bias Reduction**: Particularly effective when n is close to k
- **Bayesian Interpretation**: Can be viewed as Bayesian estimation with an informative prior

**Use Cases**:
- Portfolio optimization with limited historical data
- High-dimensional portfolios (many assets relative to observations)
- Reducing estimation error in mean returns
- Stabilizing portfolio weights across rebalancing periods

**Implementation**:
```python
shrinkage_mean = shrinkage_mean_return(rtn, stock_list)
```

**Advantages**:
- Reduces mean-squared error of return estimates
- More stable portfolio weights
- Better out-of-sample performance
- Automatic adjustment of shrinkage intensity
- Well-grounded in statistical theory

**When to Use**:
- Number of assets (k) is large relative to sample size (n)
- Sample period is short
- Portfolio weights exhibit high variability
- Looking to improve out-of-sample performance

**Mathematical Notes**:
- Shrinkage toward MVP is theoretically motivated: MVP return represents a natural "center" of the return distribution
- The shrinkage intensity formula accounts for the covariance structure, providing optimal shrinkage in mean-squared error sense
- As sample size increases, shrinkage intensity naturally decreases (s_w → 0)

---

## Choosing the Right Method

### When to Use Robust Estimation

**Use Campbell's Robust Estimation when**:
- Working with high-volatility assets (e.g., technology stocks, cryptocurrencies)
- Data contains known outliers or extreme events
- Portfolio weights from standard estimation seem unstable
- Market conditions include crashes or unusual volatility spikes
- Data quality is questionable or contains reporting errors

**Combined with optimization strategies**:
- Robust estimation works well with all optimization strategies
- Particularly valuable for MVP and MaxSharpe when outliers could distort results
- Recommended for utility-based portfolios when extreme returns affect risk perception

### When to Use Shrinkage Estimation

**Use Shrinkage Estimation when**:
- Number of assets (k) is large relative to sample size (n), e.g., k/n > 0.1
- Historical data period is short (e.g., < 2 years of daily data)
- Portfolio weights change dramatically with small data updates
- Looking to improve out-of-sample performance
- Working with newly listed assets with limited history

**Combined with optimization strategies**:
- Shrinkage is most beneficial for return-sensitive strategies (MaxSharpe, Max_util)
- Less critical for MVP which primarily depends on covariance
- Can be combined with robust estimation for maximum stability

### Combining Robust and Shrinkage Methods

**Recommended Workflow**:
1. Apply robust estimation first to handle outliers
2. Apply shrinkage to robust estimates to reduce small-sample bias
3. Use optimized estimates in portfolio construction

**Implementation Example**:
```python
# Step 1: Robust estimation to handle outliers
mean_rtn_robust, cov_m_robust = campbell_robust_est(rtn, b1=2.0, b2=1.25)

# Step 2: Shrinkage on robust estimates (if needed)
if apply_shrinkage:
    # Recreate returns with robust mean for shrinkage
    shrinkage_mean = shrinkage_mean_return(rtn, stock_list)
    # Use shrinkage mean for return-sensitive optimizations
    mean_rtn_final = shrinkage_mean
else:
    mean_rtn_final = mean_rtn_robust

# Step 3: Portfolio optimization
w = MaxSharpe(mean_rtn_final, cov_m_robust, r_daily)
```

### Optimization Strategy Selection Guide

| Strategy | Best For | Risk Profile | Key Parameter |
|----------|----------|--------------|---------------|
| **Max_util** | Custom risk preferences | Adjustable via A | Risk aversion (A) |
| **MVP** | Risk minimization | Very low | Target return (optional) |
| **MaxSharpe** | Risk-adjusted returns | Moderate | Risk-free rate (r) |
| **RtnPerRisk** | Quick implementation | Moderate | None |

### Complete Workflow Recommendation

```python
# 1. Load and inspect data
rtn, prices = load_data(stock_list)
plot_insight(cor_m, prices)

# 2. Apply robust estimation
mean_rtn, cov_m = campbell_robust_est(rtn, b1=2.0, b2=1.25)
plot_distribution(rtn, stock_list)

# 3. Apply shrinkage (if small sample or many assets)
if len(stock_list) / len(rtn) > 0.1:  # k/n ratio check
    shrinkage_mean = shrinkage_mean_return(rtn, stock_list)
    mean_rtn = shrinkage_mean

# 4. Set constraints and optimize
short_res = [1, 1, 1, 1, 1, 1]  # Short selling allowed
mean_rtn_active, cov_m_active = active(short_res, mean_rtn, cov_m)

# 5. Choose optimization strategy
w = MaxSharpe(mean_rtn_active, cov_m_active, r_daily)
w = expand_weights(w, short_res)

# 6. Analyze and visualize
mu_port, sigma_port = plot_port(w, mean_rtn, cov_m, rtn, stock_list, A)
```

## Key Functions

### Data Loading (`load_data.py`)
- `load_data(stock_list)`: Download and calculate log returns
- `plot_insight(cor_m, prices)`: Visualize correlation and prices
- `plot_distribution(rtn, stock_list)`: Plot return distributions

### Optimization (`optimization.py`)
- `cal_mean_var(weights, mean_np, cov_np)`: Calculate portfolio return and variance
- `cal_util(mu, var, A)`: Calculate utility
- `mvp(mean_rtn, cov_m, target_return=None)`: Minimum variance portfolio
- `MaxSharpe(mean_rtn, cov_m, r)`: Maximum Sharpe ratio portfolio
- `Max_util(mean_rtn, cov_m, A)`: Maximum utility portfolio
- `plot_port(w, mean_rtn, cov_m, rtn, stock_list, A)`: Comprehensive portfolio visualization

### Robust Estimation (`rubust_mean_cov.py`)
- `campbell_robust_est(rtn, b1=2.0, b2=1.25)`: Robust mean and covariance estimation

### Shrinkage (`Shrinkage.py`)
- `shrinkage_mean_return(rtn, stock_list)`: Shrinkage estimator for mean returns

### Simulation (`GBM.py`)
- `GBM_simulation(vol_annual, mu_annual, S0, T, N, M, m_show=100)`: Geometric Brownian Motion simulation

## Parameters

- **Risk Aversion (A)**: Higher values indicate greater risk aversion
- **Risk-free Rate (r)**: Used for Sharpe ratio calculation
- **Short Selling Constraints**: Binary vector indicating which assets can be shorted
- **Shrinkage**: Boolean flag to enable/disable shrinkage estimation

## Notes

- All portfolios are normalized to sum to 1
- Log returns are used throughout for computational convenience
- The project assumes daily returns for calculations
- Monte Carlo simulation uses a fixed random seed for reproducibility

## License

This project is provided as-is for educational and research purposes.

## Author

Financial Portfolio Optimization Project

