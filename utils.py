import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.base.adapters import _StatsModelsAdapter
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsSARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import yfinance as yf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dropout
from xgboost import XGBRegressor

def fetch_stock_data(ticker, start_date, end_date, freq='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    freq : str
        Frequency of data ('1d' for daily, '1wk' for weekly, '1mo' for monthly)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stock data
    """
    try:
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=freq)
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to match our app's expected format
        df = df.rename(columns={'Date': 'date', 'Close': 'value'})
        
        # Select only date and value columns
        df = df[['date', 'value']]
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


# Preprocessing
def preprocess_data(df, freq):
    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.set_index('date').sort_index()
    df.index = df.index.to_period(freq)
    df = df.loc[df.index.notnull()]
    df = df[~df.index.duplicated(keep='first')]
    return df

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    split_idx = y.index[split_point]
    y_train = y.loc[:split_idx].iloc[:-1]
    y_test = y.loc[split_idx:]
    return y_train, y_test

def create_lagged_features(series, n_lags):
    df = pd.DataFrame({"y": series})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df.dropna(inplace=True)
    return df

# Model runners
def run_naive_model(y_train, y_test, fh, **kwargs):
    forecaster = NaiveForecaster(**kwargs)
    forecaster.fit(y_train)
    
    fh_test = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh=fh_test)
    
    future_idx = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
    future_horizon = ForecastingHorizon(future_idx, is_relative=False)
    y_forecast = forecaster.predict(fh=future_horizon)
    return forecaster, y_pred, y_forecast

def run_ets_model(y_train, y_test, fh, **kwargs):
    forecaster = AutoETS(**kwargs)
    forecaster.fit(y_train)
    
    fh_test = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh=fh_test)

    future_idx = pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq)
    y_forecast = forecaster.predict(fh=ForecastingHorizon(future_idx, is_relative=False))
    return forecaster, y_pred, y_forecast

def run_arima_model(y_train, y_test, fh, **kwargs):
    forecaster = AutoARIMA(**kwargs)
    forecaster.fit(y_train)
    
    fh_test = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh=fh_test)

    future_idx = pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq)
    y_forecast = forecaster.predict(fh=ForecastingHorizon(future_idx, is_relative=False))
    return forecaster, y_pred, y_forecast

def run_rf_model(y_train, y_test, fh, **kwargs):
    n_lags = kwargs.get('n_lags', 5)
    
    df_train = create_lagged_features(y_train, n_lags)
    X_train = df_train.drop(columns="y").values
    y_train_trimmed = df_train["y"].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_trimmed)
    
    combined = pd.concat([y_train[-n_lags:], y_test])
    df_test = create_lagged_features(combined, n_lags)
    X_test = df_test.drop(columns="y").values
    y_pred = pd.Series(model.predict(X_test), index=df_test.index)
    
    forecast_input = list(y_train[-n_lags:].values)
    y_forecast_values = []
    
    for _ in range(fh):
        lag_features = np.array(forecast_input[-n_lags:]).reshape(1, -1)
        next_pred = model.predict(lag_features)[0]
        y_forecast_values.append(next_pred)
        forecast_input.append(next_pred)
    
    future_idx = pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq)
    y_forecast = pd.Series(y_forecast_values, index=future_idx)
    
    return model, y_pred, y_forecast


def build_multioutput_dataset(series, input_steps, output_steps):
    series = np.array(series)
    total_len = len(series)
    n_samples = total_len - input_steps - output_steps + 1

    X = np.array([series[i:i + input_steps] for i in range(n_samples)])
    Y = np.array([series[i + input_steps:i + input_steps + output_steps] for i in range(n_samples)])

    return X, Y

def run_rnn_model(y_train, y_test, fh, **kwargs):
    Tx = kwargs.get('n_lags', 5)
    n_epochs = kwargs.get('n_epochs', 50)
    batch_size = kwargs.get('batch_size', 32)
    units = kwargs.get('units', 64)
    strategy = kwargs.get('strategy', 'multistep')
    
    def build_rnn_model(input_shape, units=50, learning_rate=0.001, output_steps=1):
        model = Sequential()
        model.add(SimpleRNN(units, activation='relu', input_shape=input_shape))
        model.add(Dense(output_steps))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    y_train_values = y_train.drop(columns="y").values
    y_test_values = y_test.drop(columns="y").values
    
    train_index = y_train.index
    test_index = y_test.index
    
    if strategy == 'multistep':
        X_train = np.array([y_train_values[t:t+Tx] for t in range(len(y_train_values) - Tx - 1 + 1)])
        Y_train = np.array([y_train_values[t+Tx] for t in range(len(y_train_values) - Tx - 1 + 1)])
        
        X_test = np.array([y_test_values[t:t+Tx] for t in range(len(y_test_values) - Tx - 1 + 1)])
        Y_test = np.array([y_test_values[t+Tx] for t in range(len(y_test_values) - Tx - 1 + 1)])
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = build_rnn_model((Tx, 1), units=units)
    elif strategy == 'multioutput':
        X_train = np.array([y_train_values[t:t+Tx] for t in range(len(y_train_values) - Tx - fh + 1)])
        Y_train = np.array([y_train_values[t+Tx:t+Tx+fh] for t in range(len(y_train_values) - Tx - fh + 1)])
        
        X_test = np.array([y_test_values[t:t+Tx] for t in range(len(y_test_values) - Tx - fh + 1)])
        Y_test = np.array([y_test_values[t+Tx:t+Tx+fh] for t in range(len(y_test_values) - Tx - fh + 1)])
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = build_rnn_model((Tx, 1), units=units, output_steps=fh)
    else:
        raise ValueError("Invalid strategy. Choose 'multistep' or 'multioutput'.")
    
    model.fit(
        X_train, 
        Y_train, 
        epochs=n_epochs, 
        batch_size=batch_size, 
        verbose=0
    )
    test_predictions = model.predict(X_test)
    if strategy == 'multistep':
        n_preds = len(test_predictions)
        pred_idx = pd.period_range(start=test_index[Tx], periods=n_preds, freq=y_test.index.freq)
        y_pred = pd.Series(test_predictions.flatten(), index=pred_idx)
    else:
        # Extract only the first step of each forecast window
        first_steps = test_predictions[:, 0]
        # Create matching index
        pred_start_idx = test_index[Tx]
        pred_idx = pd.period_range(start=pred_start_idx, periods=len(first_steps), freq=y_test.index.freq)
        y_pred = pd.Series(first_steps, index=pred_idx)
    
    combined_values = np.concatenate([y_train_values, y_test_values])
    last_window = combined_values[-Tx:].reshape(-1)
    future_forecasts = make_future_forecasts(model, last_window, fh, strategy)
    future_idx = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
    y_forecast = pd.Series(future_forecasts.flatten(), index=future_idx)
    return model, y_pred, y_forecast
    
def make_future_forecasts(model, last_window, fh, strategy):
    if strategy == 'multistep':
        curr = last_window.copy()
        future_forecasts = []
        for _ in range(fh):
            curr_input = curr.reshape(1, curr.shape[0], 1)
            pred = model.predict(curr_input, verbose=0)
            future_forecasts.append(pred[0][0])
            curr = np.roll(curr, -1)
            curr[-1] = pred[0][0]
        return np.array(future_forecasts).reshape(-1, 1)
    elif strategy == 'multioutput':
        curr_input = last_window.reshape(1, last_window.shape[0], 1)
        future_forecasts = model.predict(curr_input)
        return future_forecasts.flatten()
    else:
        raise ValueError("Invalid strategy. Choose 'multistep' or 'multioutput'.")

def run_lstm_model(y_train, y_test, fh, **kwargs):
    n_lags = kwargs.get('n_lags', 5)
    n_epochs = kwargs.get('n_epochs', 50)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', 64)
    
    df_train = create_lagged_features(y_train, n_lags)
    X_train = df_train.drop(columns="y").values
    y_train_trimmed = df_train["y"].values
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = build_lstm_model((n_lags, 1), units, learning_rate)
    model.fit(X_train, y_train_trimmed, epochs=n_epochs, batch_size=32, verbose=0)
    
    df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
    X_test = df_test.drop(columns="y").values
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_pred = pd.Series(model.predict(X_test).flatten(), index=df_test.index)

    # Forecast future values recursively
    forecast_input = list(pd.concat([y_train, y_test]).iloc[-n_lags:].values)
    y_forecast_values = []
    for _ in range(fh):
        input_seq = np.array(forecast_input[-n_lags:]).reshape((1, n_lags, 1))
        next_val = model.predict(input_seq).flatten()[0]
        y_forecast_values.append(next_val)
        forecast_input.append(next_val)

    future_idx = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
    y_forecast = pd.Series(y_forecast_values, index=future_idx)

    return model, y_pred, y_forecast

def run_xgboost_model(y_train, y_test, fh, **kwargs):
    n_lags = kwargs.get('n_lags', 5)
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 3)
    learning_rate = kwargs.get('learning_rate', 0.1)
    
    df_train = create_lagged_features(y_train, n_lags)
    X_train = df_train.drop(columns="y").values
    y_train_trimmed = df_train["y"].values
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train_trimmed)
    
    df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
    X_test = df_test.drop(columns="y").values
    y_pred = pd.Series(model.predict(X_test), index=df_test.index)
    
    forecast_input = list(pd.concat([y_train, y_test]).iloc[-n_lags:].values)
    y_forecast_values = []
    for _ in range(fh):
        lag_features = np.array(forecast_input[-n_lags:]).reshape(1, -1)
        next_pred = model.predict(lag_features)[0]
        y_forecast_values.append(next_pred)
        forecast_input.append(next_pred)
    future_idx = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
    y_forecast = pd.Series(y_forecast_values, index=future_idx)
    return model, y_pred, y_forecast

def build_lstm_model(input_shape, units=50, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units//2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def calculate_metrics(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        # Handle division by zero in MAPE calculation
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    except Exception as e:
        # Return default values if calculation fails
        return {"MAE": float('nan'), "RMSE": float('nan'), "MAPE": float('nan')}

def rank_and_style_metrics(metrics_df, sort_by="MAPE"):
    # Sort by selected metric (lower is better)
    sorted_df = metrics_df.sort_values(by=sort_by)

    # Highlight best value with a color that works in dark mode
    def highlight_best(s):
        is_min = s == s.min()
        return ['background-color: #2d5a27' if v else '' for v in is_min]  # Dark green that works in dark mode

    styled = sorted_df.style.format("{:.2f}").apply(highlight_best, axis=0)
    return styled

MODEL_REGISTRY = {
    'Naive': run_naive_model,
    'ETS': run_ets_model,
    'ARIMA': run_arima_model,
    # 'SARIMA': run_sarima_model,
    'RandomForest': run_rf_model,
    'RNN': run_rnn_model,
    'LSTM': run_lstm_model,
    'XGBoost': run_xgboost_model,
}