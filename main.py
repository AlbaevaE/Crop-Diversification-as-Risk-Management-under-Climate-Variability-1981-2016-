from src.xray_v import MaizeYieldDataset  # если класс в xray_v.py
import matplotlib.pyplot as plt
from src.analysis import CropDiversificationAnalysis
from src.regional_masks import REGIONS 


# dataset = MaizeYieldDataset(
#     data_dir="gdhy_v1/maize",
#     start_year=1981,
#     end_year=2016
# )


# dataset.load()
# dataset.clean()


# summary = dataset.summary()
# print(summary)


# global_trend = dataset.global_mean_by_year()

# plt.figure(figsize=(10, 5))
# global_trend.plot(marker="o")
# plt.title("Global mean maize yield (1981–2016)")
# plt.xlabel("Year")
# plt.ylabel("Yield")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# delta_map = dataset.delta(1981, 2016)

# plt.figure(figsize=(12, 6))
# delta_map.plot()
# plt.title("Change in maize yield: 2016 vs 1981")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()

regions = REGIONS

analysis = CropDiversificationAnalysis(
    rice_path="./data/rice",
    maize_path="./data/maize",
    wheat_path="./data/wheat",
    soybean_path="./data/soybean"
)

regional_results = analysis.run_regional_analysis(regions)


regional_results["South_Asia"]["correlation"]
regional_results["South_Asia"]["volatility"]


corr = analysis.correlation_matrix()
print(corr)

vol = analysis.volatility_summary()
print(vol)

analysis.plot_timeseries()
