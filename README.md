# Agriculture Data Analysis

Анализ диверсификации сельскохозяйственных культур как инструмент управления рисками.

## Описание

Проект для анализа урожайности основных сельскохозяйственных культур (рис, кукуруза, пшеница, соя) за период 1981-2016 годов.

## Структура проекта

```
agriculture_data/
├── data/              # NetCDF файлы с данными урожайности
│   ├── rice/
│   ├── maize/
│   ├── wheat/
│   └── soybean/
├── src/               # Исходный код
│   ├── analysis.py           # Основной класс анализа
│   ├── data_loader.py        # Загрузка данных
│   ├── regional_masks.py     # Региональные маски
│   └── xray_v.py             # Вспомогательные функции
├── notebooks/         # Jupyter notebooks
└── main.py            # Основной скрипт запуска
```

## Установка

```bash
pip install xarray pandas matplotlib numpy
```

## Использование

```python
from src.analysis import CropDiversificationAnalysis
from src.regional_masks import REGIONS

analysis = CropDiversificationAnalysis(
    rice_path="./data/rice",
    maize_path="./data/maize",
    wheat_path="./data/wheat",
    soybean_path="./data/soybean"
)

# Региональный анализ
regional_results = analysis.run_regional_analysis(REGIONS)

# Корреляционная матрица
corr = analysis.correlation_matrix()

# Волатильность
vol = analysis.volatility_summary()

# Визуализация
analysis.plot_timeseries()
```

## Лицензия

[Укажите лицензию]
