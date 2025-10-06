# HOLT WINTERS METHOD
 

# AIM:
To implement the Holt Winters Method Model using Python.

# ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

# PROGRAM:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# Load the dataset
file_path = "gld_price_data.csv"  # Update with your actual file path
data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# Use only the GLD (Gold price) column
data['GLD'] = pd.to_numeric(data['GLD'], errors='coerce')
data = data.dropna(subset=['GLD'])

# Resample the data to monthly frequency (mean of each month)
monthly_data = data['GLD'].resample('MS').mean()

# Split the data into train and test sets (90% train, 10% test)
train_data = monthly_data[:int(0.9 * len(monthly_data))]
test_data = monthly_data[int(0.9 * len(monthly_data)):]

# Holt-Winters model with additive trend and seasonality
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12  # yearly seasonality for monthly data
).fit()

# Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# Plot the results
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title('Train, Test, and Predicted using Holt-Winters (Additive Trend/Seasonality)')
plt.show()

# Evaluate model performance
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

# Fit the model to the entire dataset and forecast the future
final_model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

forecast_predictions = final_model.forecast(steps=12)  # Forecast 12 future months

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
monthly_data.plot(legend=True, label='Original Data')
forecast_predictions.plot(legend=True, label='Forecasted Data', color='red')
plt.title('Original and Forecasted Gold Prices (Holt-Winters Additive)')
plt.show()

```

### OUTPUT:

## TEST_PREDICTION

<img width="992" height="705" alt="495479448-8a8b6356-7a7b-4ea9-8f7e-f74a0d42a9e5" src="https://github.com/user-attachments/assets/d41beba4-da61-4454-a576-2893ee661407" />

### FINAL_PREDICTION

<img width="986" height="701" alt="495479489-2b9a7afd-ccc2-4787-b163-4d1943ea7867" src="https://github.com/user-attachments/assets/5daad06e-ad8d-4cd1-918c-a45f9ad381c7" />


# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
