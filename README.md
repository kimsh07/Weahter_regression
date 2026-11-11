# Weather Regression

Hourly wind-speed prediction using observations from ASOS station 108.
All timestamps and date ranges use Korea Standard Time (KST).

## Prediction task

For each prediction time, estimate the actual wind speed exactly one hour
later. The target is `actual_wind_speed_ms`, measured in m/s.

The models use these five inputs:

1. `current_wind_speed_ms` — wind speed at the prediction time
2. `wind_speed_1h_before_ms` — wind speed one hour before
3. `wind_speed_2h_before_ms` — wind speed two hours before
4. `current_temperature_C` — temperature at the prediction time
5. `current_humidity_pct` — humidity at the prediction time

Rows requiring missing inputs, prior observations, or a future target were
omitted when preparing the hourly dataset; missing values were not filled.

## Training and evaluation

Rows were ordered by target time before splitting:

- Training target times: 2023-01-01 through 2024-11-30, inclusive
- Evaluation target times: 2024-12-01 through 2024-12-31, inclusive
- Training rows: 16,720
- Evaluation rows: 744

The December evaluation observations were not used to fit or update the
models. Negative regression predictions are clipped to zero.

| Method | December 2024 MAE (m/s) |
|---|---:|
| Current-wind persistence | 0.653495 |
| Linear regression | **0.594577** |
| Random forest | 0.607206 |

The persistence baseline predicts the next hour's wind speed as the current
wind speed. Linear regression had the lowest MAE in this evaluation.

### Linear regression

The model is ordinary least squares with an intercept:

```text
prediction = intercept + sum(feature_value * feature_coefficient)
```

Intercept: `0.5482589048186898`

| Feature | Coefficient |
|---|---:|
| `current_wind_speed_ms` | 0.5770659584651339 |
| `wind_speed_1h_before_ms` | 0.16818972287242112 |
| `wind_speed_2h_before_ms` | 0.05836686827311097 |
| `current_temperature_C` | 0.0007048468428028365 |
| `current_humidity_pct` | -0.0016086337526223238 |

### Random forest

The random forest uses scikit-learn's `RandomForestRegressor` with 100 trees,
`random_state=42`, and `n_jobs=1`. The fixed random state makes its randomized
training choices reproducible for the same data and software environment.
The model was trained on the same 16,720 rows and the same five inputs as the
linear regression.

## Model files

- `wind_model.json` — linear-regression coefficients and random-forest metadata.
- `wind_random_forest.joblib` — fitted random-forest estimator, saved with
  Joblib compression.

The source observations, prepared dataset, and prediction table are not included.
Load the fitted estimator with scikit-learn 1.9.1 and Joblib:

```python
import joblib

model = joblib.load("wind_random_forest.joblib")
```

Provide input columns in the feature order documented above. The JSON file
records the model settings and the negative-prediction policy; clip predictions
below zero to zero when applying either regression model.
