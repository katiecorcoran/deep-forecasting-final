# Author: Prof. Pedram Jahangiry
# Modified by: Greyson
# Added SARIMA, Random Forest, and RNN support
# Enhanced styling and layout (Dark Mode)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base.adapters import _StatsModelsAdapter
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsSARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam

# Custom CSS for dark mode and title styling
st.markdown("""
    <style>
    body, .css-18e3th9, .css-1d391kg, .css-1cpxqw2, .sidebar .sidebar-content {
        background-color: #0e1117 !important;
        color: #f5f5f5 !important;
    }
    .sidebar .sidebar-content {
        background-color: #1c1f26 !important;
    }
    h1 {
        font-size: 4em !important;
        text-align: center;
        color: #f5f5f5;
        margin-top: 0.5em;
        margin-bottom: 1em;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    </style>
""", unsafe_allow_html=True)

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

# Removed duplicate title to avoid rendering twice
# st.title("Time Series Forecasting App")

class SARIMAXForecaster(_StatsModelsAdapter):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__()
        self._kwargs = {"order": order, "seasonal_order": seasonal_order}

    _tags = {"python_version": None, "univariate-only": True, "requires-fh-in-fit": False}

    def _fit_forecaster(self, y, X=None):
        self._forecaster = StatsSARIMAX(
            endog=y,
            order=self._kwargs["order"],
            seasonal_order=self._kwargs["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)



def create_lagged_features(series, n_lags):
    df = pd.DataFrame({"y": series})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df.dropna(inplace=True)
    return df

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def run_forecast(y_train, y_test, model, fh, **kwargs):
    if model == 'Naive':
        strategy = kwargs.get('strategy', 'last')
        window_length = kwargs.get('window_length', None)
        forecaster = NaiveForecaster(strategy=strategy, window_length=window_length)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    elif model == 'ETS':
        forecaster = AutoETS(**kwargs)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    elif model == 'SARIMA':
        order = kwargs.get('order', (1, 1, 1))
        seasonal_order = kwargs.get('seasonal_order', (1, 1, 1, 12))
        forecaster = SARIMAXForecaster(order=order, seasonal_order=seasonal_order)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    elif model == 'RandomForest':
        n_lags = kwargs.get('n_lags', 5)
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        model_rf = RandomForestRegressor(n_estimators=100)
        model_rf.fit(X_train, y_train_trimmed)
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        y_pred = pd.Series(model_rf.predict(X_test), index=df_test.index)
        last_known = pd.concat([y_train, y_test]).iloc[-n_lags:]
        forecast_input = create_lagged_features(pd.concat([last_known, pd.Series([0]*fh)])).drop(columns="y").values[:fh]
        y_forecast = pd.Series(model_rf.predict(forecast_input), index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
    elif model == 'RNN':
        n_lags = kwargs.get('n_lags', 5)
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        model_rnn = build_rnn_model((n_lags, 1))
        model_rnn.fit(X_train, y_train_trimmed, epochs=50, verbose=0)
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = pd.Series(model_rnn.predict(X_test).flatten(), index=df_test.index)
        future_input = pd.concat([y_train, y_test]).iloc[-n_lags:].values
        y_forecast_values = []
        for _ in range(fh):
            input_seq = np.array(future_input[-n_lags:]).reshape((1, n_lags, 1))
            next_val = model_rnn.predict(input_seq).flatten()[0]
            y_forecast_values.append(next_val)
            future_input = np.append(future_input, next_val)
        y_forecast = pd.Series(y_forecast_values, index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
    else:
        raise ValueError("Unsupported model")

    return forecaster, y_pred, y_forecast

def plot_time_series(y_train, y_test, results, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    for model, result in results.items():
        y_pred = result['y_pred']
        y_forecast = result['y_forecast']
        ax.plot(y_pred.index.to_timestamp(), y_pred.values, label=f"{model} Test Predictions")
        ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label=f"{model} Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig

def main():
    if 'example_shown' not in st.session_state:
        st.session_state.example_shown = True
    st.title("Time Series Forecasting App")

    if st.session_state.example_shown:
        st.markdown("""
        ### Example Forecast Visualization
        Here's an example of what the forecast output might look like. This will disappear once you upload your own dataset.
        """)
        example_dates = pd.date_range(start="2022-01-01", periods=24, freq='M').to_period('M')
        example_values = pd.Series(np.sin(np.linspace(0, 3 * np.pi, 24)) * 100 + 200, index=example_dates)
        example_forecast = example_values[-6:] + np.random.normal(0, 10, size=6)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(example_values.index.to_timestamp(), example_values.values, label="Historical Data")
        ax.plot(example_values.index[-6:].to_timestamp(), example_forecast, label="Example Forecast")
        plt.title("Example Forecast Plot")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(fig)
    st.sidebar.header("Configuration")

    model_choices = st.sidebar.multiselect("Select model(s)", ["Naive", "ETS", "ARIMA", "SARIMA", "RandomForest", "RNN"])
    fh = st.sidebar.number_input("Forecast horizon", min_value=1, value=12)
    train_size = st.sidebar.slider("Train size (%)", 50, 95, 80) / 100

    use_example = st.sidebar.checkbox("Use example data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if use_example:
        st.session_state.example_shown = False
        df = pd.DataFrame({
            "date": pd.date_range(start="2022-01-01", periods=20, freq='M'),
            "value": [120, 130, 125, 140, 150, 160, 155, 165, 170, 180,
                       190, 200, 210, 220, 215, 225, 230, 235, 240, 250]
        })
        freq = st.sidebar.selectbox("Frequency", options=['D', 'W', 'M', 'Q', 'Y'], index=2)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date').sort_index()
        df.index = df.index.to_period(freq)
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        target = st.sidebar.selectbox("Target variable", numeric_columns)

        run_button = st.sidebar.button("Run Forecast")

        st.subheader("Preview of Data")
        st.write(df.head())

        y = df[target]
        y_train, y_test = manual_train_test_split(y, train_size)

        results = {}

        if run_button:
            for model in model_choices:
                with st.spinner(f"Running {model}..."):
                    st.write(f"### Running {model}...")
                    params = {}
                    if model in ['Naive']:
                        params['strategy'] = 'last'
                    if model in ['RandomForest', 'RNN']:
                        params['n_lags'] = 5
                    if model == 'SARIMA':
                        params['order'] = (1, 1, 1)
                        params['seasonal_order'] = (1, 1, 1, 12)
                    try:
                        forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model, fh, **params)
                        results[model] = {"y_pred": y_pred, "y_forecast": y_forecast}
                    except Exception as e:
                        st.error(f"Error in model {model}: {e}")
                params = {}
                if model in ['Naive']:
                    params['strategy'] = 'last'
                if model in ['RandomForest', 'RNN']:
                    params['n_lags'] = 5
                if model == 'SARIMA':
                    params['order'] = (1, 1, 1)
                    params['seasonal_order'] = (1, 1, 1, 12)
                try:
                    forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model, fh, **params)
                    results[model] = {"y_pred": y_pred, "y_forecast": y_forecast}
                except Exception as e:
                    st.error(f"Error in model {model}: {e}")

        if results:
            st.subheader("Forecast Comparison")
            fig = plot_time_series(y_train, y_test, results, "Forecast Results")
            st.pyplot(fig)

if __name__ == "__main__":
    main()

