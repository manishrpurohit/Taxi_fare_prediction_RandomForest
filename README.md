# Taxi Fare Prediction with Random Forest

This project predicts taxi fares from pickup/drop-off coordinates and passenger count using a Random Forest regression model. The complete analysis, training workflow, hyperparameter tuning, and a small Folium map example are in the Jupyter notebook.

## Project files

| File | Description |
| --- | --- |
| `Taxi Fare Prediction.ipynb` | Exploratory analysis, model training, GridSearchCV tuning, and map visualization. |
| `TaxiFare.csv` | Dataset containing 50,000 taxi trip records. |

## Dataset

The target is `amount` (the taxi fare). The original data contains these columns:

`unique_id`, `amount`, `date_time_of_pickup`, `longitude_of_pickup`, `latitude_of_pickup`, `longitude_of_dropoff`, `latitude_of_dropoff`, and `no_of_passenger`.

The notebook drops `unique_id` and `date_time_of_pickup` before modeling. The remaining coordinate and passenger-count columns are used as features. The dataset has no missing values according to the notebook's check.

## Workflow

1. Load `TaxiFare.csv` with pandas.
2. Inspect data types, missing values, and summary statistics.
3. Remove the ID and pickup-time columns.
4. Split the data into 80% training and 20% test sets (`random_state=42`).
5. Train a baseline `RandomForestRegressor`.
6. Tune `n_estimators`, `max_depth`, and `min_samples_split` using 5-fold `GridSearchCV`.
7. Evaluate with MAE, MSE, RMSE, and R-squared.

## Recorded results

The notebook records the following test-set performance:

| Model | MAE | MSE | RMSE | R-squared |
| --- | ---: | ---: | ---: | ---: |
| Baseline Random Forest | 2.3545 | 25.0009 | 5.0001 | 0.7311 |
| Tuned Random Forest | 2.3357 | 24.4219 | 4.9419 | 0.7373 |

Best hyperparameters found:

```python
{
    "max_depth": None,
    "min_samples_split": 10,
    "n_estimators": 150,
}
```

## Getting started

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn folium jupyter
```

3. Start Jupyter from the project directory:

```bash
jupyter notebook
```

4. Open `Taxi Fare Prediction.ipynb` and run the cells in order.

## Notes

- The raw data includes some implausible values (for example, negative fares and out-of-range coordinates). The current notebook models the data as-is; cleaning or filtering these records is a useful next improvement.
- Pickup time is currently discarded. Extracting time-based features such as hour, day of week, and month could improve the model.
- Geographic distance and route-related features are not engineered in the current workflow and are likely to be valuable additions.
