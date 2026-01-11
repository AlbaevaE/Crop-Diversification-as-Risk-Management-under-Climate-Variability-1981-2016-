import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimization for crop diversification.
    
    Implements:
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Efficient frontier generation
    """
    
    def __init__(self, returns_df):
        """
        Initialize optimizer with historical returns/yield data.
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame with crops as columns, time as index
            (should be standardized yield anomalies)
        """
        self.returns_df = returns_df
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()
        self.n_assets = len(returns_df.columns)
        
    def minimum_variance_weights(self):
        """
        Calculate minimum variance portfolio weights.
        
        Solves: min w'Σw subject to Σw = 1, w ≥ 0
        
        Returns
        -------
        dict
            Optimal weights, portfolio volatility, and portfolio return
        """
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return weights @ self.cov_matrix @ weights
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        weights = result.x
        portfolio_vol = np.sqrt(portfolio_variance(weights))
        portfolio_return = weights @ self.mean_returns
        
        return {
            'weights': pd.Series(weights, index=self.returns_df.columns),
            'volatility': portfolio_vol,
            'return': portfolio_return,
            'optimization_result': result
        }
    
    def maximum_sharpe_weights(self, risk_free_rate=0.0):
        """
        Calculate maximum Sharpe ratio portfolio weights.
        
        Solves: max (μ'w - rf) / sqrt(w'Σw) subject to Σw = 1, w ≥ 0
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate (default 0 for yield analysis)
        
        Returns
        -------
        dict
            Optimal weights, Sharpe ratio, volatility, and return
        """
        # Objective function: negative Sharpe ratio (minimize negative = maximize positive)
        def neg_sharpe_ratio(weights):
            portfolio_return = weights @ self.mean_returns
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            
            # Avoid division by zero
            if portfolio_vol == 0:
                return 1e10
            
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        weights = result.x
        portfolio_return = weights @ self.mean_returns
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        
        return {
            'weights': pd.Series(weights, index=self.returns_df.columns),
            'sharpe_ratio': sharpe,
            'volatility': portfolio_vol,
            'return': portfolio_return,
            'optimization_result': result
        }
    
    def efficient_frontier(self, n_points=50):
        """
        Generate efficient frontier points.
        
        Creates portfolios with different target returns and finds
        the minimum variance for each target.
        
        Parameters
        ----------
        n_points : int
            Number of points on the frontier
        
        Returns
        -------
        pd.DataFrame
            Frontier with columns: return, volatility, weights
        """
        # Range of target returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_vols = []
        frontier_weights = []
        
        for target_return in target_returns:
            try:
                # Minimize variance subject to target return
                def portfolio_variance(weights):
                    return weights @ self.cov_matrix @ weights
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'eq', 'fun': lambda w: w @ self.mean_returns - target_return}
                ]
                
                bounds = tuple((0, 1) for _ in range(self.n_assets))
                initial_weights = np.array([1/self.n_assets] * self.n_assets)
                
                result = minimize(
                    portfolio_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    vol = np.sqrt(portfolio_variance(result.x))
                    frontier_vols.append(vol)
                    frontier_weights.append(result.x)
                else:
                    frontier_vols.append(np.nan)
                    frontier_weights.append(np.full(self.n_assets, np.nan))
                    
            except Exception:
                frontier_vols.append(np.nan)
                frontier_weights.append(np.full(self.n_assets, np.nan))
        
        frontier_df = pd.DataFrame({
            'return': target_returns,
            'volatility': frontier_vols
        })
        
        # Add weights as separate columns
        for i, crop in enumerate(self.returns_df.columns):
            frontier_df[f'weight_{crop}'] = [w[i] for w in frontier_weights]
        
        # Remove NaN results
        frontier_df = frontier_df.dropna()
        
        return frontier_df
    
    def compare_strategies(self):
        """
        Compare equal-weighted, min-variance, and max-Sharpe portfolios.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with volatility and return for each strategy
        """
        # Equal-weighted portfolio
        equal_weights = np.array([1/self.n_assets] * self.n_assets)
        equal_return = equal_weights @ self.mean_returns
        equal_vol = np.sqrt(equal_weights @ self.cov_matrix @ equal_weights)
        
        # Min-variance portfolio
        min_var = self.minimum_variance_weights()
        
        # Max-Sharpe portfolio
        max_sharpe = self.maximum_sharpe_weights()
        
        comparison = pd.DataFrame({
            'Strategy': ['Equal-Weighted', 'Minimum Variance', 'Maximum Sharpe'],
            'Return': [equal_return, min_var['return'], max_sharpe['return']],
            'Volatility': [equal_vol, min_var['volatility'], max_sharpe['volatility']],
            'Sharpe Ratio': [
                equal_return / equal_vol if equal_vol > 0 else 0,
                min_var['return'] / min_var['volatility'] if min_var['volatility'] > 0 else 0,
                max_sharpe['sharpe_ratio']
            ]
        })
        
        return comparison
