# Predictive Analytics Using Historical Data

## Project overview
This beginner-friendly project uses historical monthly sales data to predict sales for the next six months. It demonstrates data cleaning, preprocessing, regression, evaluation, and visualization.

## Objectives
- Clean and prepare historical data.
- Train a Linear Regression model.
- Evaluate predictions using MAE, RMSE, and R².
- Forecast the next six months and visualize the trend.

## Folder structure
- `data/historical_sales.csv` — sample historical monthly sales data.
- `src/predictive_analytics.py` — full Python source code.
- `outputs/six_month_forecast.csv` — example forecast.
- `outputs/evaluation_metrics.csv` — example evaluation metrics.
- `outputs/sales_forecast.png` — forecast visualization.
- `outputs/model_evaluation.png` — actual-versus-predicted chart.
- `PROJECT_REPORT.md` — project report.

## How to run
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python src/predictive_analytics.py
   ```
5. Open the `outputs` folder to see updated CSV files and charts.

## Dataset
The included dataset is synthetic sample data created for learning and demonstration. Replace it with your own historical data if available. Keep the columns `Date` and `Sales`, with one row per month.

## Method
The project creates a chronological train/test split to avoid training on future rows when evaluating the model. A Linear Regression model learns the overall trend from month index to sales. The model is then retrained on all rows and used to estimate the following six months.

## Limitations
This simple regression model mainly captures a linear trend and does not explicitly model seasonal patterns, promotions, market changes, or unusual events. Forecasts are estimates, not guarantees. For stronger performance, compare time-series methods and use more historical data.
