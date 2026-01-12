import xarray as xr
import glob
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_crop_folder


class CropDiversificationAnalysis:
    """
    Case 3: Crop diversification as risk management
    Period: 1981–2016
    """

    def __init__(self, rice_path, maize_path, wheat_path, soybean_path, var_name="var"):
        self.var_name = var_name

        # load_crop_folder returns DataArray with name "yield", not Dataset
        self.rice = load_crop_folder(rice_path, var_name=var_name)
        self.maize = load_crop_folder(maize_path, var_name=var_name)
        self.wheat = load_crop_folder(wheat_path, var_name=var_name)
        self.soybean = load_crop_folder(soybean_path, var_name=var_name)

        # placeholders
        self.rice_mean = None
        self.maize_mean = None
        self.wheat_mean = None
        self.soybean_mean = None

        self.rice_z = None
        self.maize_z = None
        self.wheat_z = None
        self.soybean_z = None

        self.portfolio = None
        self.df = None

    # ---------- data loading ----------

    # def _load_crop(self, folder):
    #     files = sorted(glob.glob(f"{folder}/*.nc4"))

    #     if not files:
    #         raise ValueError(f"No NetCDF files found in {folder}")

    #     ds = xr.open_mfdataset(
    #         files,
    #         concat_dim="time",
    #         combine="nested",
    #         parallel=True
    #     )

    #     return ds

    # ---------- spatial aggregation ----------

    def compute_global_means(self):
        # load_crop_folder returns DataArray directly, not Dataset
        self.rice_mean = self.rice.mean(dim=("lat", "lon"))
        self.maize_mean = self.maize.mean(dim=("lat", "lon"))
        self.wheat_mean = self.wheat.mean(dim=("lat", "lon"))
        self.soybean_mean = self.soybean.mean(dim=("lat", "lon"))

        return self.rice_mean, self.maize_mean, self.wheat_mean, self.soybean_mean

    # ---------- standardization ----------

    @staticmethod
    def _standardize(da):
        return (da - da.mean("time")) / da.std("time")

    def standardize_series(self):
        if self.rice_mean is None:
            self.compute_global_means()

        self.rice_z = self._standardize(self.rice_mean)
        self.maize_z = self._standardize(self.maize_mean)
        self.wheat_z = self._standardize(self.wheat_mean)
        self.soybean_z = self._standardize(self.soybean_mean)
        return self.rice_z, self.maize_z, self.wheat_z, self.soybean_z

    # ---------- dataframe & correlation ----------

    def build_dataframe(self):
        if self.rice_z is None:
            self.standardize_series()

        self.df = pd.DataFrame(
            {
                "rice": self.rice_z.values,
                "maize": self.maize_z.values,
                "wheat": self.wheat_z.values,
                "soybean": self.soybean_z.values,
            },
            index=self.rice_z.time.values
        )

        self.df.index.name = "year"
        return self.df

    def correlation_matrix(self):
        if self.df is None:
            self.build_dataframe()

        return self.df.corr()

    # ---------- diversification portfolio ----------

    def compute_portfolio(self):
        if self.rice_z is None:
            self.standardize_series()

        self.portfolio = (self.rice_z + self.maize_z + self.wheat_z + self.soybean_z) / 4
        return self.portfolio

    def volatility_summary(self):
        if self.portfolio is None:
            self.compute_portfolio()

        return {
            "rice_std": float(self.rice_z.std().values),
            "maize_std": float(self.maize_z.std().values),
            "wheat_std": float(self.wheat_z.std().values),
            "soybean_std": float(self.soybean_z.std().values),
            "portfolio_std": float(self.portfolio.std().values),
        }


    # ---------- visualization ----------

    def plot_timeseries(self):
        if self.portfolio is None:
            self.compute_portfolio()

        plt.figure(figsize=(10, 5))

        self.rice_z.plot(label="Rice")
        self.maize_z.plot(label="Maize")
        self.wheat_z.plot(label="Wheat")
        self.soybean_z.plot(label="soybean")
        self.portfolio.plot(
            label="Diversified portfolio",
            linewidth=4,
            color="black"
        )

        plt.legend()
        plt.title("Crop diversification as risk management (1981–2016)")
        plt.xlabel("Year")
        plt.ylabel("Standardized yield anomaly")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


#---------- regional analysis ----------

    def _apply_region_mask(self, da, region):
        if region is None:
           return da

        lat_min, lat_max = region["lat"]
        lon_min, lon_max = region["lon"]
    
        lat_mask = (da.lat >= lat_min) & (da.lat <= lat_max)
    
        if lon_min <= lon_max:
            lon_mask = (da.lon >= lon_min) & (da.lon <= lon_max)
        else:
            # region crosses 0°
            lon_mask = (da.lon >= lon_min) | (da.lon <= lon_max)
    
        return da.where(lat_mask & lon_mask)


    def regional_mean(self, crop_da, region):
        masked = self._apply_region_mask(crop_da, region)
        return masked.mean(dim=("lat", "lon"))


    def analyze_region(self, region_name, region_def):
       # load_crop_folder returns DataArray directly, not Dataset
       rice_r = self.regional_mean(self.rice, region_def)
       maize_r = self.regional_mean(self.maize, region_def)
       wheat_r = self.regional_mean(self.wheat, region_def)
       soybean_r = self.regional_mean(self.soybean, region_def)
    
       # standardize
       rice_z = self._standardize(rice_r)
       maize_z = self._standardize(maize_r)
       wheat_z = self._standardize(wheat_r)
       soybean_z = self._standardize(soybean_r)
    
       portfolio = (rice_z + maize_z + wheat_z + soybean_z) / 4
    
       df = pd.DataFrame(
           {
               "rice": rice_z.values,
               "maize": maize_z.values,
               "wheat": wheat_z.values,
               "soybean": soybean_z.values,
               "portfolio": portfolio.values,
           },
           index=rice_z.time.values
       )
    
       return {
           "region": region_name,
           "correlation": df[["rice", "maize", "wheat", "soybean"]].corr(),
           "volatility": df.std(),
           "timeseries": df,
       }


    def run_regional_analysis(self, regions):
        results = {}
    
        for name, region in regions.items():
            print(f"Analyzing {name}...")
            results[name] = self.analyze_region(name, region)
    
        return results


# ---------- tail risk analysis ----------

    def value_at_risk(self, quantile=0.05):
        """
        Calculate Value at Risk (VaR) for individual crops and portfolio.
        
        VaR represents the threshold yield below which yields fall with
        probability equal to the quantile (default 5%).
        
        Parameters
        ----------
        quantile : float
            Quantile for VaR calculation (default 0.05 = 5th percentile)
        
        Returns
        -------
        dict
            VaR values for each crop and portfolio
        """
        if self.portfolio is None:
            self.compute_portfolio()
        
        return {
            "rice_var": float(self.rice_z.quantile(quantile).values),
            "maize_var": float(self.maize_z.quantile(quantile).values),
            "wheat_var": float(self.wheat_z.quantile(quantile).values),
            "soybean_var": float(self.soybean_z.quantile(quantile).values),
            "portfolio_var": float(self.portfolio.quantile(quantile).values),
        }
    
    def conditional_var(self, quantile=0.05):
        """
        Calculate Conditional Value at Risk (CVaR, also called Expected Shortfall).
        
        CVaR is the expected value conditional on being in the worst quantile%.
        It captures tail risk better than VaR.
        
        Parameters
        ----------
        quantile : float
            Quantile threshold (default 0.05 = worst 5%)
        
        Returns
        -------
        dict
            CVaR values for each crop and portfolio
        """
        if self.portfolio is None:
            self.compute_portfolio()
        
        # Calculate VaR thresholds
        var_vals = self.value_at_risk(quantile)
        
        # CVaR = mean of values below VaR threshold
        rice_cvar = float(self.rice_z.where(self.rice_z <= var_vals["rice_var"]).mean().values)
        maize_cvar = float(self.maize_z.where(self.maize_z <= var_vals["maize_var"]).mean().values)
        wheat_cvar = float(self.wheat_z.where(self.wheat_z <= var_vals["wheat_var"]).mean().values)
        soybean_cvar = float(self.soybean_z.where(self.soybean_z <= var_vals["soybean_var"]).mean().values)
        portfolio_cvar = float(self.portfolio.where(self.portfolio <= var_vals["portfolio_var"]).mean().values)
        
        return {
            "rice_cvar": rice_cvar,
            "maize_cvar": maize_cvar,
            "wheat_cvar": wheat_cvar,
            "soybean_cvar": soybean_cvar,
            "portfolio_cvar": portfolio_cvar,
        }
    
    def identify_extreme_events(self, threshold=-1.5):
        """
        Identify years with extreme negative yield shocks.
        
        Parameters
        ----------
        threshold : float
            Z-score threshold for extreme events (default -1.5 std dev)
        
        Returns
        -------
        dict
            Years where each crop experienced extreme events,
            and years where ANY crop was extreme
        """
        if self.rice_z is None:
            self.standardize_series()
        
        # Find years below threshold for each crop
        rice_extreme = self.rice_z.where(self.rice_z < threshold, drop=True).time.values
        maize_extreme = self.maize_z.where(self.maize_z < threshold, drop=True).time.values
        wheat_extreme = self.wheat_z.where(self.wheat_z < threshold, drop=True).time.values
        soybean_extreme = self.soybean_z.where(self.soybean_z < threshold, drop=True).time.values
        
        # Find years where at least one crop is extreme
        all_extreme_years = set()
        all_extreme_years.update(rice_extreme)
        all_extreme_years.update(maize_extreme)
        all_extreme_years.update(wheat_extreme)
        all_extreme_years.update(soybean_extreme)
        
        return {
            "rice_extreme_years": sorted([int(y) for y in rice_extreme]),
            "maize_extreme_years": sorted([int(y) for y in maize_extreme]),
            "wheat_extreme_years": sorted([int(y) for y in wheat_extreme]),
            "soybean_extreme_years": sorted([int(y) for y in soybean_extreme]),
            "any_crop_extreme_years": sorted([int(y) for y in all_extreme_years]),
            "threshold": threshold
        }
    
    def tail_risk_comparison(self):
        """
        Comprehensive comparison of tail risk metrics.
        
        Returns
        -------
        pd.DataFrame
            Table comparing VaR and CVaR across crops and portfolio
        """
        if self.portfolio is None:
            self.compute_portfolio()
        
        var_5 = self.value_at_risk(0.05)
        cvar_5 = self.conditional_var(0.05)
        
        comparison = pd.DataFrame({
            "Crop": ["Rice", "Maize", "Wheat", "Soybean", "Portfolio"],
            "VaR (5%)": [
                var_5["rice_var"],
                var_5["maize_var"],
                var_5["wheat_var"],
                var_5["soybean_var"],
                var_5["portfolio_var"]
            ],
            "CVaR (5%)": [
                cvar_5["rice_cvar"],
                cvar_5["maize_cvar"],
                cvar_5["wheat_cvar"],
                cvar_5["soybean_cvar"],
                cvar_5["portfolio_cvar"]
            ]
        })
        
        return comparison

# ---------- time-varying correlation analysis (Enhancement 2) ----------

    def rolling_correlation(self, window=10):
        """
        Compute rolling window correlations between all crop pairs.
        
        Parameters
        ----------
        window : int
            Rolling window size in years (default 10)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling correlations for each crop pair
        """
        if self.df is None:
            self.build_dataframe()
        
        crops = ['rice', 'maize', 'wheat', 'soybean']
        pairs = []
        for i, crop1 in enumerate(crops):
            for crop2 in crops[i+1:]:
                pairs.append((crop1, crop2))
        
        rolling_corrs = pd.DataFrame(index=self.df.index)
        
        for crop1, crop2 in pairs:
            pair_name = f"{crop1}-{crop2}"
            # Compute rolling correlation
            rolling_corrs[pair_name] = self.df[crop1].rolling(window=window).corr(self.df[crop2])
        
        return rolling_corrs
    
    def decade_comparison(self):
        """
        Compare correlation and volatility statistics across decades.
        
        Returns
        -------
        dict
            Dictionary with statistics for each decade
        """
        if self.df is None:
            self.build_dataframe()
        
        decades = {
            '1980s': (1981, 1989),
            '1990s': (1990, 1999),
            '2000s': (2000, 2009),
            '2010s': (2010, 2016)
        }
        
        results = {}
        
        for decade_name, (start, end) in decades.items():
            # Filter data for this decade
            mask = (self.df.index >= start) & (self.df.index <= end)
            decade_df = self.df[mask]
            
            if len(decade_df) < 3:
                continue
            
            # Compute correlation matrix
            corr_matrix = decade_df[['rice', 'maize', 'wheat', 'soybean']].corr()
            
            # Get average pairwise correlation (upper triangle only)
            crops = ['rice', 'maize', 'wheat', 'soybean']
            pair_corrs = []
            for i, crop1 in enumerate(crops):
                for crop2 in crops[i+1:]:
                    pair_corrs.append(corr_matrix.loc[crop1, crop2])
            avg_correlation = sum(pair_corrs) / len(pair_corrs)
            
            # Compute portfolio volatility
            portfolio = decade_df[['rice', 'maize', 'wheat', 'soybean']].mean(axis=1)
            
            results[decade_name] = {
                'years': f"{start}-{end}",
                'n_years': len(decade_df),
                'avg_pairwise_correlation': float(avg_correlation),
                'portfolio_volatility': float(portfolio.std()),
                'individual_volatility': {
                    crop: float(decade_df[crop].std()) for crop in crops
                },
                'correlation_matrix': corr_matrix
            }
        
        return results
    
    def plot_rolling_correlations(self, window=10):
        """
        Plot rolling correlations over time.
        
        Parameters
        ----------
        window : int
            Rolling window size in years (default 10)
        """
        rolling_corrs = self.rolling_correlation(window)
        
        plt.figure(figsize=(12, 6))
        
        for col in rolling_corrs.columns:
            plt.plot(rolling_corrs.index, rolling_corrs[col], label=col, linewidth=2)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Year')
        plt.ylabel('Correlation')
        plt.title(f'Rolling {window}-Year Correlations Between Crop Yields')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()