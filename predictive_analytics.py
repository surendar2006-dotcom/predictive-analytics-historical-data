"""
Predictive Analytics Using Historical Data
Beginner-friendly project: forecast monthly sales using Linear Regression.
Run: pip install -r requirements.txt
     python src/predictive_analytics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "historical_sales.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    data = data.dropna(subset=["Date", "Sales"]).sort_values("Date")
    data = data.drop_duplicates(subset=["Date"])
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.dropna(subset=["Sales"])
    data["MonthIndex"] = np.arange(len(data))

    if len(data) < 12:
        raise ValueError("Please provide at least 12 months of historical data.")

    # Chronological split: first 30 rows for training, remaining rows for testing.
    split = max(1, int(len(data) * 0.83))
    X, y = data[["MonthIndex"]], data["Sales"]
    model = LinearRegression()
    model.fit(X.iloc[:split], y.iloc[:split])
    test_predictions = model.predict(X.iloc[split:])
    actual = y.iloc[split:]

    mae = mean_absolute_error(actual, test_predictions)
    rmse = mean_squared_error(actual, test_predictions) ** 0.5
    r2 = r2_score(actual, test_predictions) if len(actual) > 1 else float("nan")
    metrics = pd.DataFrame([{
        "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)
    }])
    metrics.to_csv(OUT_DIR / "evaluation_metrics.csv", index=False)

    # Refit on all historical data and forecast the next six months.
    final_model = LinearRegression().fit(X, y)
    future_dates = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                                 periods=6, freq="MS")
    future_index = np.arange(len(data), len(data) + 6).reshape(-1, 1)
    future_values = final_model.predict(future_index)
    forecast = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predicted_Sales": np.round(future_values, 2)
    })
    forecast.to_csv(OUT_DIR / "six_month_forecast.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Sales"], marker="o", label="Historical sales")
    plt.plot(future_dates, future_values, marker="o", linestyle="--", label="Forecast")
    plt.title("Historical Sales and Six-Month Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales (sample units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sales_forecast.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(data["Date"].iloc[split:], actual, marker="o", label="Actual")
    plt.plot(data["Date"].iloc[split:], test_predictions, marker="s", label="Predicted")
    plt.title("Model Evaluation: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Sales (sample units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_evaluation.png", dpi=160)
    plt.close()

    print("Model evaluation")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.3f}")
    print("\\nSix-month forecast:")
    print(forecast.to_string(index=False))
    print(f"\\nOutput files saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
