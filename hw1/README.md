# Jetson Air diffusion in the US: a Bass-model forecast (DS 223 homework)

**Innovation:** Jetson Air (TIME Best Inventions 2025) | **Look-alike:** US air-source heat pumps | **Scope:** United States

## Structure
- `report/report.pdf` is the final report. `report/report_source.md` is the source Markdown.
- `analysis.py` fits the Bass models, runs the back-test and forecasts, and writes `img/`, `data/*.csv` and `results.json`.
- `build_report.py` fills `report_source.md` from `results.json` and renders the PDF (pandoc + wkhtmltopdf).
- `data/` holds the input data and the outputs:
  - `us_air_source_heat_pump_shipments_2001_2025.csv`: input (Statista/AHRI).
  - `bass_fit_results.csv`: fits for all windows and M choices.
  - `category_forecast_by_year.csv`: net new US heat-pump households by year, 2021-2050.
  - `jetson_forecast_by_year.csv`: Jetson Air installs by year under the three share scenarios.
- `img/` holds figures 1-5.
- `references/` holds the reference list and a register of every external number.

## Reproduce
```
pip install -r requirements.txt
python analysis.py
python build_report.py     # needs pandoc and wkhtmltopdf installed
```

## Method in four lines
1. Fit Bass (p, q, M) to US heat-pump shipments, 2009-2025 (nonlinear least squares, multi-start).
2. Convert shipments to net new heat-pump households with ratio k, from EIA RECS 2009/2015/2020 stock versus AHRI shipments.
3. Forecast net new households 2026-2050 under three k values (0.17 / 0.24 / 0.30).
4. Jetson installs = category flow x Jetson share. The share is an explicit assumption: 0.3% in 2026 rising to 0.5% / 2% / 5% by 2035.
