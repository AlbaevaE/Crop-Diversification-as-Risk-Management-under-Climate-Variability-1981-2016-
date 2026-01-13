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
    
    def maximum_sortino_weights(self, risk_free_rate=0.0, target_return=0.0):
        """
        Calculate maximum Sortino ratio portfolio weights.
        
        Solves: max (μ'w - rf) / DownsideDeviation(w) subject to Σw = 1, w ≥ 0
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate (default 0 for yield analysis)
        target_return : float
            Target return for downside deviation calculation (default 0)
        
        Returns
        -------
        dict
            Optimal weights, Sortino ratio, volatility, and return
        """
        # Calculate downside deviation
        def downside_deviation(weights):
            portfolio_returns = self.returns_df @ weights
            downside_returns = portfolio_returns[portfolio_returns < target_return] - target_return
            if len(downside_returns) == 0:
                return 1e-6 # Avoid division by zero
            return np.sqrt(np.mean(downside_returns**2))

        # Objective function: negative Sortino ratio (minimize negative = maximize positive)
        def neg_sortino_ratio(weights):
            portfolio_return = weights @ self.mean_returns
            start_dev = downside_deviation(weights)
            if start_dev == 0:
                return 1e10
            
            sortino = (portfolio_return - risk_free_rate) / start_dev
            return -sortino
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            neg_sortino_ratio,
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
        
        downside_dev = downside_deviation(weights)
        sortino = (portfolio_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        return {
            'weights': pd.Series(weights, index=self.returns_df.columns),
            'sortino_ratio': sortino,
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
        
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_vols = []
        frontier_weights = []
        
        for target_return in target_returns:
            try:
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
        
        for i, crop in enumerate(self.returns_df.columns):
            frontier_df[f'weight_{crop}'] = [w[i] for w in frontier_weights]
        
        frontier_df = frontier_df.dropna()
        
        return frontier_df
    
    def compare_strategies(self):
        """
        Compare equal-weighted, min-variance, and max-Sortino portfolios.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with volatility and return for each strategy
        """
        
        equal_weights = np.array([1/self.n_assets] * self.n_assets)
        equal_return = equal_weights @ self.mean_returns
        equal_vol = np.sqrt(equal_weights @ self.cov_matrix @ equal_weights)
        
        min_var = self.minimum_variance_weights()
        max_sortino = self.maximum_sortino_weights()
        
        comparison = pd.DataFrame({
            'Strategy': ['Equal-Weighted', 'Minimum Variance', 'Maximum Sortino'],
            'Return': [equal_return, min_var['return'], max_sortino['return']],
            'Volatility': [equal_vol, min_var['volatility'], max_sortino['volatility']],
            'Sortino Ratio': [
                np.nan, 
                np.nan,
                max_sortino['sortino_ratio']
            ]
        })
        
        return comparison
