# CA6001 Citi Bike Station Demand Project

This workspace contains a small-scale AI project on hourly Citi Bike station demand forecasting. The notebooks cover source-data acquisition, data checks and filtering, task-specific dataset construction, exploratory visualisation, lag-feature engineering, chronological evaluation, and XGBoost modelling.

The assignment-style written report is available at `docs/report.md`.

## Study design

The prediction unit is one station-hour. The future target will be the number of trips starting at a station during an hour (`pickup_count`). The current scope uses January and February 2024 and retains the 50 stations with the highest valid pickup volume in January, which is the planned training period. Using only the training period for station selection prevents future information from influencing the study population.

This restriction was chosen because it:

- preserves a continuous 60-day time series for chronological model evaluation;
- includes repeated weekday, weekend, and hour-of-day patterns;
- keeps the processed dataset small enough for an ordinary laptop;
- uses Citi Bike as the only behavioural dataset;
- avoids requiring an external borough-boundary dataset.

The scope does not support claims about annual seasonality. That limitation must be stated in the final report.

## Environment

The project targets Python 3.11 and uses `uv` for dependency management.

```bash
uv sync
uv run python -m ipykernel install --user --name ca6001-citibike --display-name "Python (CA6001 Citi Bike)"
uv run jupyter lab
```

Open `notebooks/CA6001_Citi_Bike_End_to_End.ipynb` and select the `Python (CA6001 Citi Bike)` kernel. This single notebook downloads missing archives, validates and processes trips in chunks, creates the station-hour panel, performs exploratory analysis, constructs leakage-safe lag features, applies a chronological split, trains XGBoost, and evaluates it against a weekly seasonal-naive baseline.

To execute the notebook non-interactively:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/CA6001_Citi_Bike_End_to_End.ipynb \
  --output CA6001_Citi_Bike_End_to_End.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

## Data source and local outputs

Source: [Citi Bike System Data](https://citibikenyc.com/system-data)

Official archives:

- `https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip`
- `https://s3.amazonaws.com/tripdata/202402-citibike-tripdata.zip`

The notebook creates:

- `data/raw/202401-citibike-tripdata.zip`
- `data/raw/202402-citibike-tripdata.zip`
- `data/processed/filtered_trips.parquet`
- `data/processed/station_hour_demand.parquet`
- `data/processed/station_metadata.parquet`
- `data/processed/modeling_dataset.parquet`
- `reports/figures/*.png`
- `reports/figures/05_station_activity_map.html`

Raw archives and generated outputs are ignored by Git because they can be reproduced from the notebook.

## Filtering rules

A trip is retained when:

1. its ride ID, start station ID, station name, and start timestamp are present;
2. the timestamp is parseable and falls inside the selected source month;
3. the end timestamp is parseable and later than the start timestamp;
4. duration is between 1 minute and 24 hours;
5. the start coordinates are valid;
6. the start station belongs to the top 50 stations by valid January pickup count.

Duplicate ride IDs are removed after combining the monthly archives. The 24-hour upper bound is a transparent quality-control choice rather than an official Citi Bike rule.

## Task-specific dataset

`station_hour_demand.parquet` is a balanced panel. Every selected station has one row for every hour from 2024-01-01 00:00 through 2024-02-29 23:00. Hours without a recorded pickup are explicitly represented by `pickup_count = 0`.

The table also contains calendar features derived from Citi Bike timestamps and a reproducible New York holiday calendar:

- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `month`
- `is_holiday`

`holiday_name` is retained for interpretation but should not normally be one-hot encoded in this two-month experiment because each named holiday has too few observations. The binary `is_holiday` flag is the intended model feature.

`latitude`, `longitude`, `station_name`, and `total_pickups` are descriptive station metadata. For the planned fixed-station models, use categorical `station_id` and exclude these metadata fields by default. In particular, `total_pickups` is an EDA aggregate and must not be used as a predictor. Coordinates can be tested in a separate spatial-generalisation experiment; regularised linear models require scaling, while tree models do not.

The holiday calendar is external calendar metadata, not an external demand or weather dataset. Its incremental value should later be checked with an ablation experiment.

Lagged predictors are constructed in the modelling part of the notebook so that their leakage controls can be evaluated alongside the chronological train/validation/test split.

## Modelling protocol

Notebook 02 treats the problem as rolling one-hour-ahead forecasting: when predicting hour *t*, observed demand through hour *t-1* is available. Lag features are computed separately within each station and include short-term lags (1, 2, 3, and 6 hours), daily/weekly lags (24, 48, and 168 hours), and shifted rolling statistics. All rolling windows are shifted before aggregation, so the current target never enters its own predictors.

The chronological split is:

- training: 2024-01-08 through 2024-02-07;
- validation: 2024-02-08 through 2024-02-14;
- test: 2024-02-15 through 2024-02-29.

The first seven days are used only as lag history. Hyperparameters and early stopping use the validation set; the test set is evaluated once after model selection. This protocol evaluates one-hour-ahead forecasts with observed history, not recursive multi-hour forecasts.

## Reproducibility notes

- Downloads are idempotent: an existing non-empty archive is not downloaded again.
- Processed files are replaced rather than appended, preventing duplicate records across reruns.
- The notebook reads CSV files directly from ZIP archives in chunks and does not extract multi-gigabyte intermediate CSV files.
- The official source removes service/test rides and trips shorter than 60 seconds; the notebook still validates duration because malformed records may occur.
