# Optimal Portfolio Allocation

A Python tool for calculating optimal investment portfolio allocations based on Modern Portfolio Theory. This application analyzes historical stock data, calculates risk-return characteristics, and determines the optimal portfolio weights that maximize utility for a given risk preference.

## Features
 
- **Stock Data Retrieval**: Fetches historical price data from Yahoo Finance using the yfinance API
- **Return Analysis**: Calculates and visualizes the distribution of daily returns for each stock
- **Correlation Analysis**: Computes and displays correlation coefficients across all stocks in the portfolio
- **Portfolio Optimization**: Solves for optimal asset weights that maximize investor utility based on specified risk aversion

## Theory

The optimization is based on Modern Portfolio Theory, which seeks to maximize the utility function:

$$
U = \mathbf{w}^T \boldsymbol{\mu} - \frac{A}{2} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}
$$

Where:
- $U$ = Investor's utility
- $\mathbf{w}$ = Vector of portfolio weights ($w_1, w_2, ..., w_n$)
- $\boldsymbol{\mu}$ = Vector of expected returns
- $\boldsymbol{\Sigma}$ = Covariance matrix of asset returns
- $A$ = Risk aversion coefficient

The solution provides the optimal asset allocation that balances expected returns against portfolio risk according to the investor's specific risk tolerance.

## Usage

1. Input the list of stocks you want to include in your portfolio
2. The application automatically fetches historical price data from Yahoo Finance
3. Analyze the distribution of daily returns for each stock:
   <img src="demo/dist_list.png" alt="Daily Returns Distribution" width="800" />

   
4. Review correlation relationships between all stocks:

   <img src="demo/COV.png" alt="Correlation Matrix" width="1000" />

5. Specify your risk aversion parameter (higher values indicate greater risk aversion)
6. The tool calculates, using gradient descent with Adam optimizer, the optimal portfolio allocation that maximizes your utility function.
   <img src="demo/Learning_curve.png" alt="Learning Curve and portfolio return distribution" width="500" />
- **Risk Management**: Use correlation insights to balance portfolio concentration and avoid unintended risk overlaps
- **Diversification**: Construct portfolios with low-correlation assets to reduce overall portfolio risk without sacrificing expected returns

  <img src="demo/Long_only.png" alt="Example of Long-only portfolio" width="500" />
  
- **Hedging**: Implement long-short strategies on highly correlated assets to mitigate specific risk exposures

  <img src="demo/Long_short.png" alt="Example of Long-Short portfolio" width="500" />
  
7. An interactive window displays the excess return probability. The user can move along the slider bar to choose the target return rate.
   
   <img src="demo/excess_return.png" alt="Example of excess return probability" width="500" />
## Requirements

- Python 3.12+
- yfinance
- pandas
- numpy
- matplotlib
- scipy

