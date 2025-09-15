# 📊 Optimal Portfolio Allocation using Modern Portfolio Theory

A comprehensive Python tool for calculating optimal investment portfolio allocations based on Modern Portfolio Theory. This application analyzes historical stock data, calculates risk-return characteristics, and determines optimal portfolio weights using advanced optimization algorithms including Adam, Gradient Descent, and Closed-Form solutions.

## ✨ Features

### 🎯 **Multi-Asset Portfolio Optimization**
- **9 Diversified Assets**: Technology (NVDA, AAPL, MSFT, INTC), E-commerce (AMZN), Automotive (TSLA), Retail (COST), Aerospace (BA), Cryptocurrency (BTC)
- **Sector Diversification**: Balanced exposure across different market sectors
- **Risk Management**: Long-only and long-short portfolio strategies

### 📈 **Advanced Data Analysis**
- **Hybrid Methodology**: Daily data for training (2024), monthly data for testing (2025)
- **Statistical Analysis**: Comprehensive return distribution analysis with Gaussian fitting
- **Correlation Analysis**: Full correlation matrix with heatmap visualization
- **Risk Metrics**: Volatility, Sharpe ratio, maximum drawdown calculations

### 🧠 **Multiple Optimization Algorithms**
- **Adam Optimization**: Adaptive moment estimation with momentum and learning rate adaptation
- **Gradient Descent**: Classical first-order optimization method
- **Closed-Form Solution**: Analytical solution using matrix inversion
- **Convergence Analysis**: Learning curves and optimization performance comparison

### 🎮 **Interactive Features**
- **Real-time GUI**: PyQt5-based interactive distribution explorer
- **Dynamic Sliders**: Adjust return thresholds and see probability calculations
- **Visual Analytics**: Comprehensive charts and statistical visualizations
- **Performance Tracking**: Out-of-sample testing with real market data

## 🔬 Theory

The optimization is based on **Modern Portfolio Theory**, which seeks to maximize the utility function:

$$
U = \mathbf{w}^T \boldsymbol{\mu} - \frac{A}{2} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}
$$

**Subject to constraints:**
- Budget constraint: $\sum_{i=1}^{n} w_i = 1$ (weights sum to 100%)
- Long-only constraint: $w_i \geq 0$ (no short selling)

**Where:**
- $U$ = Investor's utility function
- $\mathbf{w}$ = Vector of portfolio weights ($w_1, w_2, ..., w_n$)
- $\boldsymbol{\mu}$ = Vector of expected returns (from 2024 daily data)
- $\boldsymbol{\Sigma}$ = Covariance matrix of asset returns (from 2024 daily data)
- $A$ = Risk aversion coefficient (higher values = more risk averse)

## 🚀 Usage

### 1. **Data Collection & Processing**
```python
# The application automatically fetches historical data
# Training: 2024 daily data for optimization
# Testing: 2025 monthly data for evaluation
```

### 2. **Statistical Analysis**
Analyze the distribution of daily returns for each stock:
<img src="demo/dist_list.png" alt="Daily Returns Distribution" width="800" />

### 3. **Correlation Analysis**
Review correlation relationships between all stocks:
<img src="demo/COV.png" alt="Correlation Matrix" width="1000" />

### 4. **Portfolio Optimization**
Choose your optimization algorithm and risk aversion parameter:
- **Adam**: Advanced optimization with adaptive learning rates
- **Gradient Descent**: Classical optimization method
- **Closed-Form**: Analytical solution

<img src="demo/Learning_curve.png" alt="Learning Curve and portfolio return distribution" width="1000" />

### 5. **Portfolio Allocation Results**
View optimal asset allocations and performance metrics:

**Long-Only Portfolio Example:**
<img src="demo/Long_only.png" alt="Example of Long-only portfolio" width="500" />

**Long-Short Portfolio Example:**
<img src="demo/Long_short.png" alt="Example of Long-Short portfolio" width="500" />

### 6. **Interactive Analysis**
Explore return probabilities with the interactive GUI:
<img src="demo/excess_return.png" alt="Example of excess return probability" width="500" />

### 7. **Out-of-Sample Testing**
Evaluate portfolio performance using real 2025 market data:
- **Expected vs Actual Returns**: Compare predictions with reality
- **Risk-Adjusted Performance**: Sharpe ratio and volatility analysis
- **Individual Asset Contributions**: See which positions drove performance

## 📊 Key Results

### **Portfolio Performance (Example)**
- **Expected Annual Return (2024)**: 7.64%
- **Actual Monthly Return (2025)**: 9.67%
- **Outperformance**: +2.03% above expectations
- **Sharpe Ratio**: 0.17 (risk-adjusted performance)

### **Investment Scenario ($10,000)**
- **Initial Investment**: $10,000.00
- **Expected Value**: $10,764.24
- **Actual Value**: $10,966.89
- **Profit**: $202.65 (1.9% above expectations)

## 🛠️ Installation & Requirements

### **System Requirements**
- Python 3.12+
- Windows 10/11 (for PyQt5 GUI)
- 4GB+ RAM recommended

### **Dependencies**
```bash
pip install yfinance pandas numpy matplotlib scipy PyQt5 seaborn
```

### **Quick Start**
1. Clone the repository
2. Install dependencies
3. Run the Jupyter notebook
4. Follow the interactive analysis

## 📈 Methodology

### **Training Phase (2024)**
- **Data Source**: Daily stock prices from Yahoo Finance
- **Processing**: Daily percentage changes scaled by 21 (monthly approximation)
- **Optimization**: Full covariance matrix calculation for robust statistical modeling
- **Algorithms**: Adam, Gradient Descent, or Closed-Form optimization

### **Testing Phase (2025)**
- **Data Source**: January 2025 monthly data
- **Processing**: Simple monthly return rates for realistic evaluation
- **Validation**: Out-of-sample performance testing
- **Metrics**: Real investment return calculations

## 🎯 Use Cases

### **Individual Investors**
- **Portfolio Construction**: Build diversified portfolios based on risk tolerance
- **Asset Allocation**: Optimize weight distribution across different asset classes
- **Risk Management**: Understand portfolio risk characteristics and correlations

### **Financial Professionals**
- **Research & Analysis**: Test portfolio optimization strategies
- **Client Advisory**: Demonstrate portfolio construction methodologies
- **Risk Assessment**: Analyze correlation patterns and diversification benefits

### **Educational Purposes**
- **Learning MPT**: Understand Modern Portfolio Theory concepts
- **Algorithm Comparison**: Compare different optimization methods
- **Statistical Analysis**: Learn about return distributions and correlations

## ⚠️ Important Notes

### **Risk Disclaimers**
- **Past Performance**: Historical data does not guarantee future results
- **Market Risk**: All investments carry inherent market risk
- **Model Limitations**: Optimization models are based on historical assumptions
- **Single Month Testing**: Limited test period may not reflect long-term performance

### **Data Considerations**
- **Market Hours**: Data reflects trading hours and market conditions
- **Corporate Actions**: Stock splits and dividends are automatically adjusted
- **Liquidity**: Some assets may have limited trading volume
- **Currency**: All data in USD unless otherwise specified

## 🔧 Technical Details

### **Optimization Algorithms**
- **Adam**: β₁=0.9, β₂=0.999, ε=1e-8, learning rate=0.00005
- **Gradient Descent**: Learning rate=0.00001, max iterations=500,000
- **Closed-Form**: Matrix inversion with pseudo-inverse fallback

### **Statistical Methods**
- **Return Calculation**: Log returns for mathematical properties
- **Covariance**: Sample covariance matrix from historical data
- **Distribution Fitting**: Gaussian distribution overlay for normality testing
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility

## 📚 References

- **Modern Portfolio Theory**: Markowitz, H. (1952). Portfolio Selection. Journal of Finance
- **Adam Optimization**: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization
- **Portfolio Optimization**: Boyd, S., & Vandenberghe, L. (2004). Convex Optimization

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Algorithm optimizations

## 📞 Support

For questions, issues, or suggestions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include system information and error messages

---

**Disclaimer**: This tool is for educational and research purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.
