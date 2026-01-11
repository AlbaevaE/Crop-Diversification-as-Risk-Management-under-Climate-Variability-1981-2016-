# Crop Diversification as Risk Management (1981–2016)

## Motivation
Climate variability increases volatility of crop yields, posing risks to food security.
This project investigates whether crop diversification reduces systemic yield risk,
globally and across vulnerable regions.

## Research Questions
- How correlated are yields of rice, maize, wheat, and soybean?
- Does diversification reduce volatility compared to monocropping?
- Do diversification benefits differ by region?

## Data
- The global dataset of historical yields for major crops 1981–2016 (https://doi.pangaea.de/10.1594/PANGAEA.909132)
- Period: 1981–2016
- Crops: rice, maize, wheat, soybean

## Methodology
1. Stack yearly NetCDF files into spatio-temporal cubes
2. Compute global and regional yield anomalies
3. Standardize time series to compare volatility
4. Build diversified crop portfolios
5. Compare variance reduction across regions

## Key Findings
- Crop yields show moderate global correlation
- Diversification reduces volatility globally
- Benefits are region-specific and weaker in monoculture-dominated regions

## Tools
Python, xarray, pandas, matplotlib

## Relevance
This analysis shows that while global crop yields have increased steadily since 1981,
inter-annual variability remains high and crop-specific.

By constructing an equal-weighted crop portfolio, we demonstrate that diversification
consistently reduces yield volatility both globally and regionally.

These results suggest that crop diversification should be considered a core component
of climate adaptation and food security strategies.

## Project structure

```
agriculture_data/
├── data/              # NetCDF files with yeield data
│   ├── rice/
│   ├── maize/
│   ├── wheat/
│   └── soybean/
├── src/               # Source code
│   ├── analysis.py           # Main analysis class
│   ├── data_loader.py        # Data loading
│   ├── regional_masks.py     # Regional masks
│   └── xray_v.py             # Helper functions
├── notebooks/         # Jupyter notebooks
└── main.py            # Main script
```

## Installation

```bash
pip install xarray pandas matplotlib numpy
```

## Usage

```python
from src.analysis import CropDiversificationAnalysis
from src.regional_masks import REGIONS

analysis = CropDiversificationAnalysis(
    rice_path="./data/rice",
    maize_path="./data/maize",
    wheat_path="./data/wheat",
    soybean_path="./data/soybean"
)

# Regional analysis
regional_results = analysis.run_regional_analysis(REGIONS)

# Correlation matrix
corr = analysis.correlation_matrix()

# Volatility
vol = analysis.volatility_summary()

# Visualization
analysis.plot_timeseries()
```

## License
MIT
