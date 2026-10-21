# Hourly Citi Bike Station Demand Forecasting with XGBoost

**CA6001 – AI Algorithms Fundamentals & Application**  
**Group members:** *[Insert names and student IDs]*  
**Submission date:** 28 October 2026

## Abstract

Shared bicycle systems must place a limited fleet across many stations while demand changes by location and time. Poor allocation can leave users unable to find a bicycle and can increase the operator's rebalancing cost. This project develops a machine-learning solution for forecasting the number of Citi Bike trips beginning at a station during the next hour. To keep the work reproducible on a personal computer, the study uses January and February 2024 and the 50 stations with the greatest valid January pickup volume.

The original trip records were validated and converted into a balanced station-hour panel. Short-term demand lags, daily and weekly lags, shifted rolling statistics, cyclical time variables, station identity, weekend status, and a New York holiday indicator were used as predictors. A weekly seasonal-naive method provided an operationally meaningful baseline, while an XGBoost Poisson regressor represented the nonlinear AI solution. The data were divided chronologically, with a separate validation period used for early stopping and a final test period that was not used for training or model selection.

On the test set, XGBoost achieved an MAE of 2.483 pickups per station-hour, an RMSE of 3.729, WAPE of 34.17%, and an $R^2$ of 0.739. Compared with the weekly baseline, test MAE fell by approximately 33.6% and RMSE by approximately 35.0%. The strongest predictors were demand one hour earlier and demand at the same station and hour one week earlier. These results demonstrate useful short-term predictive performance, but they apply only to high-volume stations during a two-month winter period and assume that recent observed demand is available at prediction time.

## 1. Background and motivation

Citi Bike is a station-based bicycle-sharing system serving New York City and nearby areas. Customers begin and end trips at stations distributed across the service area. Demand is strongly dependent on location, hour of day, weekday, and recent conditions. This creates an operational problem: a station may have too few bicycles to satisfy departures or too few docks to accept returns.

An accurate short-term pickup forecast can support decisions about vehicle rebalancing, staff deployment, and station monitoring. The practical decision is not simply whether demand will be “high” or “low”; operators need an estimate of the number of pickups expected at each station. Therefore, this project treats demand forecasting as a supervised count-regression problem.

The work addresses the main requirements of the [CA6001 assignment brief](./CA6001%20Assignment%20Brief.pdf): defining a real problem, analysing an appropriate public dataset, explaining and applying AI algorithms, comparing methods, implementing a working solution, and evaluating it with suitable metrics.

## 2. Task definition and research questions

The unit of analysis is one station during one clock hour. For station $s$ and hour $t$, the target is

$$
y_{s,t} = \text{number of valid Citi Bike trips starting at station }s\text{ during hour }t.
$$

The operational task is to estimate $y_{s,t}$ one hour ahead. When the forecast is produced, demand through hour $t-1$ is assumed to have been observed. This is a rolling one-hour-ahead forecast rather than a recursive forecast many hours into the future.

The study investigates four questions:

1. Do recent, daily, and weekly demand patterns contain useful predictive information?
2. Can a nonlinear gradient-boosted tree model outperform a weekly seasonal-naive method?
3. Which predictors contribute most strongly to the fitted XGBoost model?
4. At which hours and stations does the model make its largest errors?

The intended output is a station-level forecast that could inform operational planning. The model does not predict whether a station will become empty because the historical trip data do not contain continuous bicycle inventory or dock-availability histories.

## 3. Dataset and project scope

### 3.1 Source and fields

The project uses official [Citi Bike System Data](https://citibikenyc.com/system-data). The January and February 2024 archives were downloaded from the official trip-data storage:

- `202401-citibike-tripdata.zip`
- `202402-citibike-tripdata.zip`

Together, the local archives occupy approximately 783.7 MB and contain five CSV files. A trip record includes a ride identifier, bicycle type, start and end timestamps, start and end station identifiers and names, coordinates, and membership type. The official source states that service/test rides and trips shorter than 60 seconds have already been removed.

The two consecutive months preserve hourly and weekly cycles while limiting storage and processing requirements. The 50 most active stations were selected using valid January pickups only. January belongs to the training-era data, so February validation and test outcomes did not influence station selection.

### 3.2 Scope limitations introduced by sampling

This design is intentionally manageable but narrows the conclusions:

- January and February represent winter conditions and cannot establish annual seasonality.
- High-volume stations are over-represented, so performance may not transfer to quieter or newly opened stations.
- Station selection is fixed; the project does not evaluate prediction for an unseen station.
- Weather, special events, service alerts, and live inventory are not included.

These restrictions trade external validity for a reproducible experiment that can run on an ordinary personal computer.

## 4. Data processing and exploratory analysis

### 4.1 Memory-efficient acquisition

Downloads are idempotent: an archive with the expected byte size is reused, while an incomplete download is written to a temporary file before replacement. CSV members are read directly from each ZIP archive in chunks of 250,000 rows. This avoids extracting or loading the full multi-million-row dataset into memory.

### 4.2 Sequential validation and filtering

The two archives contained 4,009,586 records. A record was retained when:

1. ride ID, start station ID, start station name, start time, and end time were present;
2. timestamps could be parsed;
3. the start timestamp fell inside its declared source month;
4. duration was between one minute and 24 hours;
5. start latitude and longitude were valid; and
6. the start station belonged to the 50 stations selected from January.

After required-field checks, 4,006,560 records remained. The source-month check retained 4,005,919, and the duration and coordinate checks retained 4,004,717 valid candidate records. Filtering to the selected stations and removing duplicate ride IDs produced 503,407 trip records. No duplicate ride IDs were found within the selected records.

### 4.3 Construction of the station-hour panel

Each retained trip contributes one pickup to its start station and start hour. Observed pickups were grouped by station and hour. A complete Cartesian product of 50 stations and all 1,440 hours from 1 January through 29 February was then created. Missing combinations were assigned `pickup_count = 0`, because omitting them would incorrectly remove genuine zero-demand periods.

The resulting panel has 72,000 unique station-hour rows, no missing values, and no duplicate station-time keys. Approximately 17.9% of station-hours have zero pickups. Mean demand is 6.99 pickups per station-hour, and the maximum is 73.

### 4.4 Initial patterns

Demand exhibits clear time structure. Weekdays have strong morning and evening peaks, with the highest mean demand around the evening commute. Weekend demand rises later and is spread more evenly across daytime hours.

![Average demand by weekday and hour](../reports/figures/03_weekday_hour_heatmap.png)

Activity also differs substantially between stations. `W 21 St & 6 Ave` was the busiest selected station across the two-month sample, with 17,893 valid pickups. The interactive station map is available at [`05_station_activity_map.html`](../reports/figures/05_station_activity_map.html).

## 5. Feature engineering

All lag and rolling features were calculated separately within each station after sorting by timestamp.

### 5.1 Short-term demand

The features `lag_1h`, `lag_2h`, `lag_3h`, and `lag_6h` describe recent demand. They are intended to capture persistence, sudden increases, and short-term station-specific conditions.

### 5.2 Long-term demand

The features `lag_24h`, `lag_48h`, and `lag_168h` represent demand one day, two days, and one week earlier. The weekly lag is particularly important because it matches both hour of day and weekday.

### 5.3 Rolling statistics

Means and standard deviations were calculated over the previous 3, 6, 24, and 168 hours. Every window was shifted by one hour before aggregation:

$$
\text{rolling mean}_{s,t,w} = \frac{1}{w}\sum_{i=1}^{w} y_{s,t-i}.
$$

This ordering prevents the current target $y_{s,t}$ from entering its own feature vector. The first 168 hours for each station were used as history and excluded from modelling, leaving 63,600 complete model-ready rows.

### 5.4 Calendar and station features

Hour of day and day of week were converted to sine and cosine variables to represent their cyclical nature. `is_weekend` and `is_holiday` were included as binary indicators. The holiday flag was generated from the US/New York subdivision calendar; `holiday_name` was retained for interpretation but not used as a high-cardinality model input.

Station ID was one-hot encoded using the same 50 categories in every split. Raw latitude and longitude were retained for mapping but excluded from the default model. For this fixed-station task, one-hot station identity represents station-specific effects more directly. The full-period `total_pickups` field was also excluded because it is a descriptive aggregate that could leak future demand information.

## 6. Experimental design

### 6.1 Chronological split

Random splitting would mix future and past observations and produce an unrealistically optimistic estimate. The data were therefore separated chronologically:

| Period | Dates | Rows | Purpose |
|---|---|---:|---|
| Lag history | 1–7 January | 8,400 | Initialise 168-hour features only |
| Training | 8 January–7 February | 37,200 | Fit model parameters |
| Validation | 8–14 February | 8,400 | Early stopping and model selection |
| Test | 15–29 February | 18,000 | Final unbiased evaluation |

![Chronological data split](../reports/figures/07_data_split.png)

Validation and test lag features use actual earlier observations. Thus, the evaluation represents a system that refreshes its one-hour forecast after each completed hour. A deployment requiring a 24-hour forecast at once would require recursive or direct multi-horizon evaluation and cannot claim these results.

### 6.2 Weekly seasonal-naive baseline

The baseline prediction is

$$
\hat{y}^{\text{weekly}}_{s,t} = y_{s,t-168}.
$$

This is a stronger comparison than a global mean because it automatically reflects station identity, weekday, and hour. A machine-learning model is useful only if it improves on this simple operational rule.

### 6.3 XGBoost model

XGBoost constructs an additive ensemble of decision trees. At boosting iteration $k$, a new tree is fitted to improve the loss of the existing ensemble:

$$
\hat{y}^{(k)}_i = \hat{y}^{(k-1)}_i + \eta f_k(x_i),
$$

where $\eta$ is the learning rate and $f_k$ is a regression tree. Trees can model nonlinear thresholds and interactions, such as a different relationship between recent demand and the evening peak at different stations.

The model used a Poisson count objective, which is appropriate for non-negative counts and produces non-negative predictions. Histogram-based tree construction reduced computational cost. The principal settings were:

| Parameter | Value |
|---|---:|
| Maximum estimators | 1,500 |
| Learning rate | 0.03 |
| Maximum depth | 6 |
| Minimum child weight | 8 |
| Row subsampling | 0.85 |
| Column subsampling | 0.85 |
| L1 regularisation | 0.05 |
| L2 regularisation | 2.0 |
| Early-stopping patience | 75 iterations |

Validation MAE selected iteration 410. Early stopping limits unnecessary tree growth, although it does not replace evaluation on an untouched test set.

![XGBoost training and validation curve](../reports/figures/08_training_curve.png)

## 7. Evaluation metrics

Four complementary metrics were calculated.

Mean absolute error is the primary measure and has a direct operational interpretation:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|.
$$

Root mean squared error gives greater weight to large misses:

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}.
$$

Weighted absolute percentage error normalises total absolute error by total demand and remains defined for individual zero-demand rows:

$$
\text{WAPE} = \frac{\sum_i |y_i-\hat{y}_i|}{\sum_i |y_i|}\times 100\%.
$$

$R^2$ is reported as a supplementary measure of explained variance. It should not replace error measures expressed in the original pickup-count units.

## 8. Results

| Split | Model | MAE | RMSE | WAPE | $R^2$ |
|---|---|---:|---:|---:|---:|
| Validation | Weekly seasonal naive | 3.7448 | 5.7255 | 46.56% | 0.4918 |
| Validation | XGBoost | **2.7823** | **4.0952** | **34.60%** | **0.7400** |
| Test | Weekly seasonal naive | 3.7396 | 5.7369 | 51.46% | 0.3827 |
| Test | XGBoost | **2.4830** | **3.7290** | **34.17%** | **0.7392** |

On the untouched test set, XGBoost reduced MAE by approximately 33.6% and RMSE by approximately 35.0% relative to the weekly baseline. Its WAPE was 17.29 percentage points lower. The similar validation and test $R^2$ values provide no evidence of a major generalisation collapse over these two February periods.

![Test error comparison](../reports/figures/09_model_comparison.png)

Across the test period, actual demand totalled 130,815 pickups and XGBoost predicted 130,279, an aggregate underprediction of approximately 0.4%. A small aggregate bias does not mean every station-hour is accurate, but it suggests that positive and negative errors are reasonably balanced overall.

The aggregate time-series plot shows that XGBoost follows repeated daily cycles more closely than the weekly baseline. It nevertheless smooths or misses some unusually high peaks.

![Observed and predicted test demand](../reports/figures/10_test_predictions.png)

## 9. Model interpretation and error analysis

Gain-based feature importance identified `lag_1h` as the strongest predictor, followed by `lag_168h`. Hour-of-day encoding, the three-hour rolling mean, and `lag_24h` were also important. This supports the project hypothesis that both very recent demand and repeated daily/weekly patterns are relevant.

![XGBoost feature importance](../reports/figures/12_feature_importance.png)

Feature importance describes how the fitted trees used predictors; it does not establish causal effects. For example, `is_holiday` appearing in the fitted model does not prove that holidays cause a specific change in demand. A model trained without the holiday feature is needed for an ablation comparison.

Errors are largest during high-activity commuting periods. Test MAE was 4.865 pickups at 17:00 and 4.479 at 18:00, compared with an overall MAE of 2.483. At 17:00 the mean bias was -0.615, indicating modest underprediction, while at 18:00 the bias was +0.861. The morning peak at 08:00 also had an above-average MAE of 3.692.

Station-level accuracy also varies. The busiest station, `W 21 St & 6 Ave`, had the largest station-level test MAE at 3.678. This is expected partly because high-volume stations offer more opportunity for large absolute errors. Operational deployment should therefore monitor both overall accuracy and station-specific performance rather than relying on a single average.

## 10. Comparison of approaches

The weekly seasonal-naive method is transparent, requires no fitting, and works well when behaviour repeats exactly from one week to the next. However, it cannot combine recent changes with several time scales and fails when the previous week was unusual.

XGBoost combines short lags, long lags, rolling variation, calendar variables, and station effects through nonlinear tree interactions. It substantially improves the observed error metrics, but it is more computationally expensive and less directly interpretable. It also requires monitoring and retraining when behaviour changes.

The comparison demonstrates the benefit of the AI method over a credible baseline, but it is not exhaustive. A regularised linear or Poisson regression would provide an additional interpretable algorithmic comparison, while a random forest would help determine whether sequential boosting is important. These are appropriate extensions for a broader model-comparison study.

## 11. Limitations, ethics, and responsible use

The main limitations are:

1. **Short seasonal coverage.** Two winter months cannot support claims about summer or annual demand.
2. **Station-selection bias.** Results primarily represent busy established stations.
3. **One-hour-ahead assumption.** The evaluation uses actual recent observations and does not measure recursive long-horizon performance.
4. **Unobserved explanatory variables.** Weather, events, transit disruptions, station capacity, and rebalancing activity are absent.
5. **Observed versus latent demand.** Recorded pickups may be lower when a station has no bicycles, so trip starts do not always equal unconstrained customer demand.
6. **Holiday definition.** The New York calendar includes legal or commemorative holidays that may not correspond to universal workplace closure.
7. **Limited algorithm comparison.** XGBoost is compared with a strong baseline, but additional learned algorithms would strengthen the conclusions.

The published dataset contains trip-level operational records but no direct personal identifiers. The analysis does not attempt to reconstruct individual identities or combine trips with personal information. Results should be used for aggregate operational planning rather than surveillance or user profiling. Any redistribution must follow the [Citi Bike Data Sharing Policy](https://citibikenyc.com/data-sharing-policy).

## 12. Conclusion

This project designed and implemented a reproducible AI workflow for hourly Citi Bike station demand forecasting. Chunked processing reduced more than four million trip records to a balanced station-hour panel that can be handled on a personal computer. Leakage-safe short-term, daily, and weekly features were combined with calendar and station variables in an XGBoost Poisson regressor.

XGBoost achieved a test MAE of 2.483 pickups per station-hour and improved materially over the weekly seasonal-naive baseline. The result supports the use of nonlinear machine learning for short-term station demand forecasting, especially when both recent and weekly demand histories are available. The strongest next steps are model-ablation experiments, comparison with an interpretable regression model, evaluation over a complete year, and integration of inventory data if the operational goal changes from pickup forecasting to empty-station risk prediction.

## Reproducibility

The complete workflow is contained in [`CA6001_Citi_Bike_End_to_End.ipynb`](../notebooks/CA6001_Citi_Bike_End_to_End.ipynb). Python 3.11 dependencies are defined in `pyproject.toml` and locked in `uv.lock`. Running `uv sync` followed by execution of the notebook reproduces processed data, model artefacts, metrics, and figures.

## References

1. Citi Bike. “System Data.” <https://citibikenyc.com/system-data>.
2. Citi Bike. “Data Sharing Policy.” <https://citibikenyc.com/data-sharing-policy>.
3. XGBoost Developers. “XGBoost Python API Reference.” <https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html>.
4. Python Holidays. “API Reference.” <https://holidays.readthedocs.io/en/latest/api/>.