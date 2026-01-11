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