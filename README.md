# Time Series Forecasting App

This web app allows users to upload and forecast time series data using a variety of classical and machine learning models. It's built to be flexible, supporting multiple data sources and giving users the ability to configure models and view forecast results in an interactive dashboard.

## Features

- Upload your own CSV time series data
- Use built-in sample datasets:
  - Energy output data (preloaded)
  - Yahoo Finance data (downloaded using a ticker)
- Select from a variety of models:
  - Naive, Drift, ETS, ARIMA, SARIMA
  - Random Forest, RNN, LSTM
- Choose forecasting horizon and model hyperparameters
- Visualize forecast results alongside actual data

## Supported Data Sources

### 1. CSV Upload

Upload your own time series CSV file. Requirements:
- Must include a date column (e.g., `Date`)
- Must include a target variable (e.g., `y`)
- Optional: Include additional features for supervised models

### 2. Energy Output CSV

A built-in dataset showing daily energy output from a renewable energy source. This dataset is included for quick testing and exploration of model capabilities.

### 3. Yahoo Finance

Enter a stock ticker (e.g., `AAPL`, `GOOGL`) to fetch historical stock data using the `yfinance` API. The app will:
- Pull `Close` prices over a user-defined date range
- Fill missing business days with forward fill
- Preprocess and forecast future values

## How It Works

1. Select a dataset
2. Configure your model
   - Choose lags, forecast horizon, and model type
   - Tune hyperparameters (e.g., epochs, learning rate)
3. Run the model
4. View forecasts
   - Interactive plots of past data and forecasts
   - Optionally download results

## Tech Stack

- Streamlit for the user interface
- pandas, numpy, scikit-learn, TensorFlow for data processing and modeling
- sktime and statsmodels for classical forecasting
- yfinance for real-time stock data

## Folder Structure

/app |-- main.py |-- model_utils.py |-- data/ |-- energy.csv |-- README.md
