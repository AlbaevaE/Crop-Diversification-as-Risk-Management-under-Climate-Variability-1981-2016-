# Crop Diversification as Risk Management (1981–2016)

## Motivation
Climate variability increases the volatility of crop yields, posing significant risks to global food security. This project investigates whether crop diversification can serve as an effective risk management strategy, aiming to reduce systemic yield risk both globally and across vulnerable regions.

## Research Questions
1. **Correlation:** How correlated are yields of major crops (rice, maize, wheat, soybean)?
2. **Diversification:** Does a diversified crop portfolio reduce volatility compared to monocropping?
3. **Regional Differences:** Do diversification benefits vary by geography?
4. **Optimization:** Can we construct optimal crop portfolios that minimize risk or maximize stability?
5. **Temporal Dynamics:** Are crop correlations changing over time due to climate change?
6. **Tail Risk:** How well does diversification reduce exposure to extreme yield failures (VaR / CVaR)?

## Key Findings

### 1. Tail Risk Protection
Diversification significantly mitigates extreme downside risk.
*   **Value at Risk (VaR 95%)**: The diversified portfolio's VaR is **-1.31** standard deviations, which is **23% less severe** than the worst-performing single crop (Soybean at -1.70).
*   **Conditional VaR (CVaR 95%)**: The expected shortfall during extreme events is **-1.93** for the portfolio, substantially better than Wheat's **-2.55**.
*   *Source: [`PortfolioOptimizer.value_at_risk`](src/portfolio_optimization.py) and [`notebooks/07_tail_risk_analysis.ipynb`](notebooks/07_tail_risk_analysis.ipynb)*

### 2. Optimization converges to Diversification
Advanced optimization strategies reinforce the value of simple diversification for this dataset.
*   The **Maximum Sortino Ratio** optimization converges to an **Equal-Weighted strategy** (approx. 25% allocation to each crop).
*   This suggests that given the standardized nature of yield anomalies, equal exposure provides a robust barrier against downside volatility without over-fitting to historical winners.
*   *Source: [`PortfolioOptimizer.maximum_sortino_weights`](src/portfolio_optimization.py) and [`notebooks/05_optimal_portfolios.ipynb`](notebooks/05_optimal_portfolios.ipynb)*

### 3. Temporal Dynamics & Correlation
Contrary to the assumption that systemic climate risk always increases correlations, yield correlations have shown a decreasing trend in this specific dataset.
*   **Average Pairwise Correlation**: Decreased from **0.83** in the 1980s to **0.51** in the 2010s.
*   **Portfolio Volatility**: Correspondingly dropped from **0.55** (1980s) to **0.26** (2010s), indicating that the benefits of diversification have potentially **strengthened** over time.
*   *Source: [`CropDiversificationAnalysis.decade_comparison`](src/analysis.py) and [`notebooks/06_temporal_dynamics.ipynb`](notebooks/06_temporal_dynamics.ipynb)*

## Data
- **Source:** Global dataset of historical yields for major crops (1981–2016) [PANGAEA Data](https://doi.pangaea.de/10.1594/PANGAEA.909132)
- **Period:** 1981–2016
- **Crops:** Rice, Maize, Wheat, Soybean
- **Format:** NetCDF spatio-temporal cubes

## Analytical Framework & Mathematical Models

This project employs advanced statistical and econometric models to quantify risk and diversification benefits.

### 1. Standardization (Z-Scores)
Method used in **`src.analysis.CropDiversificationAnalysis.standardize_series`** and **`notebooks/01_data_exploration.ipynb`**.

To make yields comparable across different crops and regions, all yield data is standardized into "yield anomalies" (Z-scores). This removes the unit differences (tonnes/hectare) and isolates the volatility signal.

**Formula:**
$$Z_{i,t} = \frac{Y_{i,t} - \mu_i}{\sigma_i}$$

Where:
- $Y_{i,t}$ is the yield of crop $i$ in year $t$
- $\mu_i$ is the long-term mean yield of crop $i$
- $\sigma_i$ is the standard deviation of yield for crop $i$

**Logic:** A Z-score of -2.0 indicates a severe failure (2 standard deviations below normal), regardless of whether it's Rice or Wheat.

### 2. Modern Portfolio Theory (MPT)
Method used in **`src.portfolio_optimization.PortfolioOptimizer`** and **`notebooks/05_optimal_portfolios.ipynb`**.

I apply Markowitz's Modern Portfolio Theory to agriculture. Instead of financial assets, I treat crops as assets in a portfolio to minimize yield volatility (risk).

**Objective Functions:**

*   **Minimum Variance Portfolio:** Finds the crop weights ($w$) that minimize total portfolio volatility.
    *   Function: `minimum_variance_weights()`
    $$\min \sigma_p^2 = w^T \Sigma w$$
    Subject to: $\sum w_i = 1, w_i \ge 0$ (no short selling)

*   **Maximum Sortino Ratio:** Finds the "efficiency" sweet spot—maximizing yield stability per unit of downside risk.
    *   Function: `maximum_sortino_weights()`
    $$\max S_p = \frac{R_p - R_f}{\sigma_{d}}$$
    Where $R_p$ is expected return (mean yield anomaly), $R_f$ is the risk-free rate (assumed 0), and $\sigma_{d}$ is the downside deviation.

**Analysis:** This allows us to move beyond simple "equal diversification" to scientifically optimal crop planting strategies.

### 3. Tail Risk Analysis (VaR & CVaR)
Method used in **`src.analysis.CropDiversificationAnalysis`** (`value_at_risk`, `conditional_var`) and **`notebooks/07_tail_risk_analysis.ipynb`**.

Standard deviation assumes normal distribution, but agricultural risks often have "fat tails" (extreme failure events). I use downside risk metrics to measure protection against disasters.

*   **Value at Risk (VaR):** The threshold loss level that will not be exceeded with $(1-\alpha)$ confidence.
    $$P(Z_p \le VaR_{\alpha}) = \alpha$$
    *Example:* VaR(5%) = −1.5 means that, over the specified time horizon, 95% of yield outcomes are above −1.5 standard deviations, while 5% fall below this threshold.

*   **Conditional Value at Risk (CVaR / Expected Shortfall):** The average loss *given* that an extreme event has occurred.
    $$CVaR_{\alpha} = E[Z_p | Z_p \le VaR_{\alpha}]$$
    *Logic:* "If a 1-in-20 year disaster happens, how bad will it be on average?"

### 4. Interpretation of Correlation Dynamics
Method used in **`src.analysis.CropDiversificationAnalysis.rolling_correlation`** and **`notebooks/06_temporal_dynamics.ipynb`**.

I employ rolling window analysis to test the hypothesis that climate change is synchronizing crop failures.

**Model:**
$$\rho_{t} = \text{Corr}(Z_{i, \tau}, Z_{j, \tau}) \quad \text{for } \tau \in [t-w, t]$$
Where $w$ is the window size (e.g., 10 years).

**Analysis:** Increasing positive correlations ($\rho \to 1$) implies diversification is becoming *less* effective. Decreasing correlations implies it is becoming *more* effective.

## Project Structure

```
agriculture_data/
├── data/                      # Raw NetCDF yield data
│   ├── rice/
│   ├── maize/
│   ├── wheat/
│   └── soybean/
├── src/                       # Source code modules
│   ├── analysis.py            # Core class: CropDiversificationAnalysis
│   ├── portfolio_optimization.py # MPT class: PortfolioOptimizer
│   ├── regional_masks.py      # Regional definitions
│   └── data_loader.py         # Utility for loading NetCDF
├── notebooks/                 # Jupyter Notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_global_trends.ipynb
│   ├── 03_diversification_analysis.ipynb
│   ├── 04_regional_risk_analysis.ipynb
│   ├── 05_optimal_portfolios.ipynb
│   ├── 06_temporal_dynamics.ipynb
│   └── 07_tail_risk_analysis.ipynb
└── main.py                    # Entry point script
```

## Classes and Modules

### `src.analysis.CropDiversificationAnalysis`
The primary class for handling yield data.
- **Data Loading:** Reads and processes NetCDF files.
- **Analysis:** Implements Z-score standardization, rolling correlations, VaR/CVaR, and regional masking.

### `src.portfolio_optimization.PortfolioOptimizer`
Implements Modern Portfolio Theory algorithms using `scipy.optimize`.
- **Efficient Frontier:** Generates the curve of optimal risk-return portfolios.
- **Optimization:** Solves quadratic programming problems for Min Variance and Max Sortino ratios.

## Analysis Notebooks

**01_data_exploration.ipynb:** Initial data inspection, visualizing crop distributions and yield histories. 
**02_global_trends.ipynb:** Analysis of long-term global yield trends, identifying growth vs. stagnation. 
**03_diversification_analysis.ipynb:** Core analysis of the "Equal-Weight" portfolio vs. individual crops. Demonstrates basic volatility reduction. 
**04_regional_risk_analysis.ipynb:** Breakdown of diversification benefits by specific geographic regions (e.g., North America, East Asia). 
**05_optimal_portfolios.ipynb:** Uses MPT to find scientifically optimal crop mixes that outperform equal weighting. 
**06_temporal_dynamics.ipynb:** Investigates how crop correlations evolve over time. Includes rolling window analysis.
**07_tail_risk_analysis.ipynb:** Focuses on extreme events using Value at Risk (VaR) and Expected Shortfall (CVaR).

## Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install xarray pandas matplotlib numpy scipy netcdf4
   ```

2. **Basic Usage (Python):**
   ```python
   from src.analysis import CropDiversificationAnalysis
   
   # Initialize analysis
   analysis = CropDiversificationAnalysis(
       rice_path="./data/rice",
       maize_path="./data/maize",
       wheat_path="./data/wheat",
       soybean_path="./data/soybean"
   )
   
   # Compute Optimal Portfolio
   optimizer = analysis.get_optimizer()
   min_var = optimizer.minimum_variance_weights()
   print("Optimal Low-Risk Weights:", min_var['weights'])
   ```

## License
MIT
