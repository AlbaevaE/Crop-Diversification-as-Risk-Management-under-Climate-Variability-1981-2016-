import xarray as xr
import numpy as np
from pathlib import Path


class MaizeYieldDataset:
    def __init__(self, data_dir, start_year=1981, end_year=2016):
        """
        data_dir   : gdhy_v1/maize
        start_year : 1981
        end_year   : 2016
        """
        self.data_dir = Path(data_dir)
        self.start_year = start_year
        self.end_year = end_year
        self.years = list(range(start_year, end_year + 1))

        self.ds = None  # сюда загрузим xarray Dataset

    # -------------------------
    # 1. Загрузка данных
    # -------------------------
    def load(self):
        """
        Загружает все nc4 файлы и объединяет их в 3D Dataset
        """
        pattern = str(self.data_dir / "yield_*.nc4")

        self.ds = xr.open_mfdataset(
            pattern,
            combine="nested",
            concat_dim="time"
        )

        # Назначаем координату времени (годы)
        self.ds = self.ds.assign_coords(time=self.years)

        return self.ds

    # -------------------------
    # 2. Очистка данных
    # -------------------------
    def clean(self):
        """
        Заменяет FillValue на NaN
        """
        if self.ds is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        var = self.ds["var"]

        if "_FillValue" in var.attrs:
            fill_value = var.attrs["_FillValue"]
            self.ds["var"] = var.where(var != fill_value)

        return self.ds

    # -------------------------
    # 3. Sanity checks
    # -------------------------
    def summary(self):
        """
        Быстрая проверка данных
        """
        if self.ds is None:
            raise RuntimeError("Dataset not loaded.")

        return {
            "time_range": (int(self.ds.time.min()), int(self.ds.time.max())),
            "lat_range": (float(self.ds.lat.min()), float(self.ds.lat.max())),
            "lon_range": (float(self.ds.lon.min()), float(self.ds.lon.max())),
            "min": float(self.ds["var"].min(skipna=True)),
            "max": float(self.ds["var"].max(skipna=True)),
            "nan_percent": float(np.isnan(self.ds["var"]).mean() * 100),
        }

    # -------------------------
    # 4. Аналитические методы
    # -------------------------
    def global_mean_by_year(self):
        """
        Средняя урожайность по миру для каждого года
        """
        return self.ds["var"].mean(dim=("lat", "lon"))

    def mean_map(self):
        """
        Средняя карта за все годы
        """
        return self.ds["var"].mean(dim="time")

    def delta(self, year_start, year_end):
        """
        Разница между двумя годами
        """
        return (
            self.ds["var"].sel(time=year_end)
            - self.ds["var"].sel(time=year_start)
        )

    # -------------------------
    # 5. Экспорт
    # -------------------------
    def to_dataframe(self):
        """
        Осторожно: большой DataFrame (~250k * years)
        """
        return self.ds["var"].to_dataframe(name="yield").reset_index()
