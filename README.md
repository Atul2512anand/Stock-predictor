# NIFTY 50 Price Forecast with Linear Regression

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)

A compact time-series experiment that uses a univariate linear regression on NIFTY 50 closing prices to forecast the next five closes.

## Overview

The notebook loads one year of daily NIFTY 50 data (4 June 2024 – 4 June 2025), keeps the `Date` and `Close` columns, and frames the problem as:

> Given the current closing price, predict the closing price five trading rows ahead.

It is an educational demonstration of autoregressive forecasting with `scikit-learn`, not a trading system. The notebook intentionally uses one feature, one model, and a short forecast horizon so the mechanics are easy to inspect.

## Dataset

| Item | Value |
|---|---|
| File | `NIFTY 50-04-06-2024-to-04-06-2025.csv` |
| Source | NSE India (manually downloaded) |
| Rows | 248 daily observations |
| Raw columns | `Date`, `Open`, `High`, `Low`, `Close`, `Shares Traded`, `Turnover (₹ Cr)` |
| Columns used | `Date`, `Close` |

The raw CSV contains a UTF-8 BOM and trailing spaces in its headers; the notebook strips them before processing.

## Method

1. Load the CSV and normalise the column names with `df.columns.str.strip()`.
2. Keep only `Date` and `Close`.
3. Parse dates with `format="%d-%b-%Y"`, sort chronologically, and reset the index.
4. Build the supervised learning problem:
   - Feature `X` = scaled current `Close`
   - Label `y` = `Close.shift(-5)` (the close five rows later)
   - Drop the five rows with no label
5. Hold out 20% of the rows for testing (`random_state=0`).
6. Fit `sklearn.linear_model.LinearRegression`.
7. Predict the last five feature rows to produce a five-step forecast.
8. Plot the historical series and the forecast, then print the test R² and predicted values.

### Reproducing the Core Logic

```python
df["Label"] = df["Close"].shift(-5)
X = preprocessing.scale(df[["Close"]])
X_lately = X[-5:]
X = X[:-5]
df.dropna(inplace=True)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LinearRegression().fit(X_train, y_train)
score = model.score(X_test, y_test)
forecast = model.predict(X_lately)
```

## Results

| Metric | Value |
|---|---:|
| Test R² | **0.6968** |
| Forecast horizon | 5 rows |
| Forecast values | 24,676.06, 24,741.04, 24,674.66, 24,647.36, 24,507.96 |

The notebook labels the R² value as "Model Accuracy"; R² is a goodness-of-fit measure, not percentage accuracy.

### Forecast dates

The notebook creates forecast dates with:

```python
pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
```

This uses calendar days and can include weekends. NIFTY does not trade on weekends, so the plotted dates should be treated as illustrative rather than as the next five trading sessions.

## Getting Started

### Google Colab

1. Open `stock_predic.ipynb` in Colab.
2. Run the first cell and upload `NIFTY 50-04-06-2024-to-04-06-2025.csv`.
3. Run the remaining cells.

### Local Jupyter

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook stock_predic.ipynb
```

Skip the `google.colab.files.upload()` cell and make sure the CSV path in `pd.read_csv(...)` points to the included file.

## Project Structure

```text
Stock-predictor/
├── stock_predic.ipynb                         # 10-cell notebook
├── NIFTY 50-04-06-2024-to-04-06-2025.csv      # Raw daily data
├── Screenshots/                               # Notebook output captures
└── README.md
```

## Notes and Limitations

- **One feature only:** the model uses the current close to predict a future close; it ignores Open, High, Low, volume, turnover, and external factors.
- **Random split:** rows are split randomly rather than chronologically. For time-series evaluation, use a chronological split or `TimeSeriesSplit`.
- **Scaling leakage:** `preprocessing.scale()` is fitted on the full series before the train/test split. Fit the scaler on the training data only.
- **Calendar-day horizon:** forecast dates include weekends; use trading-day offsets for a realistic schedule.
- **No uncertainty estimates:** the output is a single point forecast with no prediction interval.
- **Not financial advice:** the notebook is an educational exercise and is not suitable for trading decisions.

## Data Credit

Data sourced from [NSE India](https://www.nseindia.com) and manually exported for this project. Refer to NSE's terms for reuse.

## Author

**Atul Anand**  
B.Tech–M.Tech (Cyber Security), National Forensic Sciences University  
GitHub: [@Atul2512anand](https://github.com/Atul2512anand)
