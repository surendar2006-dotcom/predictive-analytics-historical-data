# Project Report: Predictive Analytics Using Historical Data

## 1. Introduction
Predictive analytics uses past data, statistical methods, and machine-learning techniques to estimate future outcomes. This project forecasts future monthly sales from historical sales data. It demonstrates a practical workflow that includes data preparation, model training, evaluation, and visualization.

## 2. Objectives
- Clean and preprocess historical data.
- Build a regression model to estimate future sales.
- Measure model performance using standard evaluation metrics.
- Visualize historical trends and future predictions.

## 3. Dataset
The project contains 36 months of sample monthly sales data from January 2023 to December 2025. The dataset includes two main columns: `Date` and `Sales`. The dataset is synthetic and is intended for educational demonstration; it is not actual company sales information.

## 4. Data Preprocessing
The program parses dates, removes rows with missing dates or sales, sorts observations chronologically, removes duplicate dates, converts sales to numeric values, and creates a numerical month index. A chronological split is used so that earlier observations train the model and later observations test it.

## 5. Algorithm and Methodology
Linear Regression is used because it is simple to understand and provides a baseline for learning the relationship between time and sales. The model learns a line that minimizes squared prediction errors on the training data. After evaluation, a final model is trained on the complete dataset and predicts the next six months.

## 6. Evaluation
The model is evaluated on the held-out final portion of the historical dataset. The example run produced:
- Mean Absolute Error (MAE): 190.71
- Root Mean Squared Error (RMSE): 217.22
- R-squared (R²): 0.237

MAE represents the average absolute difference between actual and predicted values. RMSE penalizes larger errors more strongly. R² describes how well predictions explain variation in the test data; with a small test set, it should be interpreted cautiously.

## 7. Results and Visualization
The project produces a six-month forecast CSV file and a chart showing historical sales alongside forecast values. A second chart compares actual and predicted values in the test period. These visuals help communicate the direction of the trend and the model's limitations.

## 8. Limitations and Future Improvements
The sample dataset is synthetic. Linear Regression captures a general linear trend but does not explicitly model seasonality, promotions, pricing, holidays, or sudden market changes. Future improvements could include more real observations, seasonal features, Random Forest regression, and time-series methods such as ARIMA. Model performance should be checked before using forecasts for business decisions.

## 9. Conclusion
This project demonstrates a complete beginner-level predictive analytics workflow. It shows how historical data can be cleaned, used to train a regression model, evaluated with metrics, and converted into a forecast with visualizations. The results are educational estimates rather than guaranteed future sales.

## 10. Tools Used
Python, pandas, NumPy, scikit-learn, and Matplotlib.
