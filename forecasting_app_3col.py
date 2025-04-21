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
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsSARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

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

def create_lagged_features(series, n_lags):
    df = pd.DataFrame({"y": series})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df.dropna(inplace=True)
    return df

def build_rnn_model(input_shape, units=50, learning_rate=0.001):
    model = Sequential()
    model.add(SimpleRNN(units, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_lstm_model(input_shape, units=50, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units//2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def run_forecast(y_train, y_test, model, fh, **kwargs):
    forecaster = None  # Initialize forecaster as None
    
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
        # Extract ETS parameters
        error = kwargs.get('error', 'add')
        trend = kwargs.get('trend', 'add')
        seasonal = kwargs.get('seasonal', 'add')
        seasonal_periods = kwargs.get('sp', 12)
        
        # Check if we have enough data for seasonal components
        if seasonal is not None and len(y_train) < 2 * seasonal_periods:
            # If not enough data, use non-seasonal model
            st.warning(f"Not enough data for seasonal components. Using non-seasonal ETS model.")
            seasonal = None
        
        # Create ETS model with correct parameters
        forecaster = AutoETS(
            error=error,
            trend=trend,
            seasonal=seasonal,
            sp=seasonal_periods if seasonal is not None else None
        )
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    elif model == 'ARIMA':
        # For AutoARIMA, we'll use a simpler approach with fixed parameters
        # Extract ARIMA parameters
        p = kwargs.get('p', 1)
        d = kwargs.get('d', 1)
        q = kwargs.get('q', 1)
        
        # Create a simple ARIMA model instead of AutoARIMA
        from statsmodels.tsa.arima.model import ARIMA
        model_arima = ARIMA(y_train, order=(p, d, q))
        model_fit = model_arima.fit()
        
        # Make predictions on test data
        y_pred = pd.Series(model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1), index=y_test.index)
        
        # Generate forecast
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        y_forecast = pd.Series(model_fit.forecast(steps=fh), index=future_dates)
        
        # Store the fitted model as forecaster
        forecaster = model_fit
    elif model == 'SARIMA':
        # Extract SARIMA parameters
        order = kwargs.get('order', (1, 1, 1))
        seasonal_order = kwargs.get('seasonal_order', (1, 1, 1, 12))
        
        # Create and fit the SARIMA model directly without using the adapter
        model_sarima = StatsSARIMAX(
            endog=y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model_sarima.fit(disp=False)
        
        # Make predictions on test data
        y_pred = pd.Series(model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1), index=y_test.index)
        
        # Generate forecast
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        y_forecast = pd.Series(model_fit.forecast(steps=fh), index=future_dates)
        
        # Store the fitted model as forecaster
        forecaster = model_fit
    elif model == 'RandomForest':
        n_lags = kwargs.get('n_lags', 5)
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        
        # Create lagged features for training data
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        
        # Train the Random Forest model
        model_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model_rf.fit(X_train, y_train_trimmed)
        forecaster = model_rf  # Store the model as forecaster
        
        # Create lagged features for test data
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        y_pred = pd.Series(model_rf.predict(X_test), index=df_test.index)
        
        # Generate forecast
        last_known = pd.concat([y_train, y_test]).iloc[-n_lags:]
        forecast_input = create_lagged_features(pd.concat([last_known, pd.Series([0]*fh)]), n_lags).drop(columns="y").values[:fh]
        y_forecast = pd.Series(model_rf.predict(forecast_input), index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
    elif model == 'RNN':
        n_lags = kwargs.get('n_lags', 5)
        units = kwargs.get('units', 50)
        learning_rate = kwargs.get('learning_rate', 0.001)
        epochs = kwargs.get('epochs', 50)
        
        # Create lagged features for training data
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Build and train the RNN model
        model_rnn = build_rnn_model((n_lags, 1), units=units, learning_rate=learning_rate)
        model_rnn.fit(X_train, y_train_trimmed, epochs=epochs, verbose=0)
        forecaster = model_rnn  # Store the model as forecaster
        
        # Create lagged features for test data
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = pd.Series(model_rnn.predict(X_test).flatten(), index=df_test.index)
        
        # Generate forecast
        future_input = pd.concat([y_train, y_test]).iloc[-n_lags:].values
        y_forecast_values = []
        for _ in range(fh):
            input_seq = np.array(future_input[-n_lags:]).reshape((1, n_lags, 1))
            next_val = model_rnn.predict(input_seq).flatten()[0]
            y_forecast_values.append(next_val)
            future_input = np.append(future_input, next_val)
        y_forecast = pd.Series(y_forecast_values, index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
    elif model == 'LSTM':
        n_lags = kwargs.get('n_lags', 10)
        units = kwargs.get('units', 50)
        learning_rate = kwargs.get('learning_rate', 0.001)
        epochs = kwargs.get('epochs', 50)
        
        # Create lagged features for training data
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Build and train the LSTM model
        model_lstm = build_lstm_model((n_lags, 1), units=units, learning_rate=learning_rate)
        model_lstm.fit(X_train, y_train_trimmed, epochs=epochs, verbose=0)
        forecaster = model_lstm  # Store the model as forecaster
        
        # Create lagged features for test data
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = pd.Series(model_lstm.predict(X_test).flatten(), index=df_test.index)
        
        # Generate forecast
        future_input = pd.concat([y_train, y_test]).iloc[-n_lags:].values
        y_forecast_values = []
        for _ in range(fh):
            input_seq = np.array(future_input[-n_lags:]).reshape((1, n_lags, 1))
            next_val = model_lstm.predict(input_seq).flatten()[0]
            y_forecast_values.append(next_val)
            future_input = np.append(future_input, next_val)
        y_forecast = pd.Series(y_forecast_values, index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
    elif model == 'XGBoost':
        n_lags = kwargs.get('n_lags', 10)
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 5)
        learning_rate = kwargs.get('learning_rate', 0.1)
        
        # Create lagged features for training data
        df_train = create_lagged_features(y_train, n_lags)
        X_train = df_train.drop(columns="y").values
        y_train_trimmed = df_train["y"].values
        
        # Train the XGBoost model
        model_xgb = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror'
        )
        model_xgb.fit(X_train, y_train_trimmed)
        forecaster = model_xgb  # Store the model as forecaster
        
        # Create lagged features for test data
        df_test = create_lagged_features(pd.concat([y_train[-n_lags:], y_test]), n_lags)
        X_test = df_test.drop(columns="y").values
        y_pred = pd.Series(model_xgb.predict(X_test), index=df_test.index)
        
        # Generate forecast
        last_known = pd.concat([y_train, y_test]).iloc[-n_lags:]
        forecast_input = create_lagged_features(pd.concat([last_known, pd.Series([0]*fh)]), n_lags).drop(columns="y").values[:fh]
        y_forecast = pd.Series(model_xgb.predict(forecast_input), index=pd.period_range(start=y_test.index[-1]+1, periods=fh, freq=y_train.index.freq))
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

def main():
    st.title("Time Series Forecasting App")
    
    st.sidebar.header("Configuration")

    # Create two columns in the sidebar for model selection and data source
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.subheader("Models")
        model_choices = st.multiselect("Select model(s)", ["Naive", "ETS", "ARIMA", "RandomForest", "RNN", "LSTM", "XGBoost"])
        fh = st.number_input("Forecast horizon", min_value=1, value=12)
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100
    
    with col2:
        st.subheader("Data Source")
        data_source = st.radio("Choose data source", ["Upload CSV", "Yahoo Finance"])
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            date_range = st.date_input(
                "Select date range",
                value=(datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.now()
            )
            freq = st.selectbox("Frequency", options=['1d', '1wk', '1mo'], index=0)
            
            if st.button("Fetch Stock Data"):
                with st.spinner(f"Fetching data for {ticker}..."):
                    df = fetch_stock_data(ticker, date_range[0].strftime('%Y-%m-%d'), 
                                         date_range[1].strftime('%Y-%m-%d'), freq)
                    if df is not None:
                        st.session_state.stock_data = df
                        st.success(f"Successfully fetched data for {ticker}")
                    else:
                        st.error(f"Failed to fetch data for {ticker}")

    # Model-specific parameters
    st.sidebar.header("Model Parameters")
    
    # Naive model parameters
    if "Naive" in model_choices:
        st.sidebar.subheader("Naive Model")
        naive_strategy = st.sidebar.selectbox("Naive Strategy", ["last", "mean", "drift"])
        naive_window = st.sidebar.number_input("Window Length", min_value=1, value=1)
    
    # ETS model parameters
    if "ETS" in model_choices:
        st.sidebar.subheader("ETS Model")
        ets_error = st.sidebar.selectbox("Error Type", ["add", "mul"])
        ets_trend = st.sidebar.selectbox("Trend Type", ["add", "mul", None])
        ets_seasonal = st.sidebar.selectbox("Seasonal Type", ["add", "mul", None])
        ets_seasonal_periods = st.sidebar.number_input("Seasonal Periods", min_value=1, value=12)
    
    # ARIMA model parameters
    if "ARIMA" in model_choices:
        st.sidebar.subheader("ARIMA Model")
        arima_p = st.sidebar.number_input("ARIMA p (AR order)", min_value=0, value=1)
        arima_d = st.sidebar.number_input("ARIMA d (Difference order)", min_value=0, value=1)
        arima_q = st.sidebar.number_input("ARIMA q (MA order)", min_value=0, value=1)
    
    # Random Forest model parameters
    if "RandomForest" in model_choices:
        st.sidebar.subheader("Random Forest Model")
        rf_n_estimators = st.sidebar.number_input("Number of Trees", min_value=10, value=100)
        rf_max_depth = st.sidebar.number_input("Max Depth", min_value=1, value=10)
        rf_n_lags = st.sidebar.number_input("Number of Lags", min_value=1, value=5)
    
    # RNN model parameters
    if "RNN" in model_choices:
        st.sidebar.subheader("RNN Model")
        rnn_units = st.sidebar.number_input("RNN Units", min_value=10, value=50)
        rnn_learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
        rnn_epochs = st.sidebar.number_input("Training Epochs", min_value=10, value=50)
        rnn_n_lags = st.sidebar.number_input("Number of Lags", min_value=1, value=5)
    
    # LSTM model parameters
    if "LSTM" in model_choices:
        st.sidebar.subheader("LSTM Model")
        lstm_units = st.sidebar.number_input("LSTM Units", min_value=10, value=50)
        lstm_learning_rate = st.sidebar.number_input("LSTM Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
        lstm_epochs = st.sidebar.number_input("LSTM Training Epochs", min_value=10, value=50)
        lstm_n_lags = st.sidebar.number_input("LSTM Number of Lags", min_value=1, value=10)
    
    # XGBoost model parameters
    if "XGBoost" in model_choices:
        st.sidebar.subheader("XGBoost Model")
        xgb_n_estimators = st.sidebar.number_input("XGBoost Number of Trees", min_value=10, value=100)
        xgb_max_depth = st.sidebar.number_input("XGBoost Max Depth", min_value=1, value=5)
        xgb_learning_rate = st.sidebar.number_input("XGBoost Learning Rate", min_value=0.01, value=0.1, format="%.2f")
        xgb_n_lags = st.sidebar.number_input("XGBoost Number of Lags", min_value=1, value=10)

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"]) if data_source == "Upload CSV" else None

    # Process data based on user selection
    if data_source == "Upload CSV" and uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display file information
            st.subheader("File Information")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            st.write("Columns:", ", ".join(df.columns.tolist()))
            
            # Check if the file has the required columns
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            
            # Provide guidance on file format
            with st.expander("File Format Requirements"):
                st.markdown("""
                ### Required Format
                Your CSV file should have:
                
                1. **A date column**: Contains dates in a format like YYYY-MM-DD, MM/DD/YYYY, etc.
                2. **At least one numeric column**: Contains the values you want to forecast
                
                ### Example Format
                ```
                date,value
                2020-01-01,100
                2020-01-02,105
                2020-01-03,102
                ...
                ```
                """)
            
            # Select date column with guidance
            if date_columns:
                st.write("Potential date columns found:", ", ".join(date_columns))
                date_col = st.sidebar.selectbox("Select date column", df.columns, index=df.columns.get_loc(date_columns[0]) if date_columns else 0)
            else:
                st.warning("No columns with 'date' or 'time' in the name were found. Please select the date column manually.")
                date_col = st.sidebar.selectbox("Select date column", df.columns)
            
            # Convert date column to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # Check for NaT values (failed conversions)
                nat_count = df[date_col].isna().sum()
                if nat_count > 0:
                    st.warning(f"{nat_count} dates could not be parsed. These rows will be excluded.")
                    df = df.dropna(subset=[date_col])
            except Exception as e:
                st.error(f"Error converting date column: {str(e)}")
                st.info("Please ensure your date column contains valid dates.")
                return
            
            # Set date as index
            df = df.set_index(date_col).sort_index()
            
            # Select frequency
            freq = st.sidebar.selectbox("Frequency", options=['D', 'W', 'M', 'Q', 'Y'], index=2)
            df.index = df.index.to_period(freq)
            
            # Select target variable with guidance
            if numeric_columns:
                st.write("Numeric columns found:", ", ".join(numeric_columns))
                target = st.sidebar.selectbox("Target variable", numeric_columns, index=0)
            else:
                st.error("No numeric columns found in the uploaded file. Please upload a file with numeric data.")
                return
            
            # Display data preview
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Display data statistics
            st.subheader("Data Statistics")
            st.write(df[target].describe())
            
            # Check for missing values
            missing_values = df[target].isna().sum()
            if missing_values > 0:
                st.warning(f"Found {missing_values} missing values in the target column. These will be handled automatically.")
            
            # Check for sufficient data
            if len(df) < 10:
                st.error("Not enough data points for forecasting. Please upload a file with at least 10 data points.")
                return
            
            run_button = st.sidebar.button("Run Forecast")
            
            y = df[target]
            y_train, y_test = manual_train_test_split(y, train_size)
            
            results = {}
            
            if run_button:
                for model in model_choices:
                    with st.spinner(f"Running {model}..."):
                        st.write(f"### Running {model}...")
                        params = {}
                        
                        # Set model-specific parameters
                        if model == 'Naive':
                            params['strategy'] = naive_strategy
                            params['window_length'] = naive_window
                        elif model == 'ETS':
                            params['error'] = ets_error
                            params['trend'] = ets_trend
                            params['seasonal'] = ets_seasonal
                            params['sp'] = ets_seasonal_periods  # Use 'sp' instead of 'seasonal_periods'
                        elif model == 'ARIMA':
                            params['p'] = arima_p
                            params['d'] = arima_d
                            params['q'] = arima_q
                        elif model == 'RandomForest':
                            params['n_estimators'] = rf_n_estimators
                            params['max_depth'] = rf_max_depth
                            params['n_lags'] = rf_n_lags
                        elif model == 'RNN':
                            params['units'] = rnn_units
                            params['learning_rate'] = rnn_learning_rate
                            params['epochs'] = rnn_epochs
                            params['n_lags'] = rnn_n_lags
                        elif model == 'LSTM':
                            params['units'] = lstm_units
                            params['learning_rate'] = lstm_learning_rate
                            params['epochs'] = lstm_epochs
                            params['n_lags'] = lstm_n_lags
                        elif model == 'XGBoost':
                            params['n_estimators'] = xgb_n_estimators
                            params['max_depth'] = xgb_max_depth
                            params['learning_rate'] = xgb_learning_rate
                            params['n_lags'] = xgb_n_lags
                            
                        try:
                            forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model, fh, **params)
                            results[model] = {"y_pred": y_pred, "y_forecast": y_forecast}
                        except Exception as e:
                            st.error(f"Error in model {model}: {e}")
                
                if results:
                    st.subheader("Forecast Comparison")
                    fig = plot_time_series(y_train, y_test, results, "Forecast Results")
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please check your file format and try again.")
            
    elif data_source == "Yahoo Finance" and 'stock_data' in st.session_state:
        df = st.session_state.stock_data
        
        # Select frequency for forecasting
        freq_map = {'1d': 'D', '1wk': 'W', '1mo': 'M'}
        freq = freq_map.get(freq, 'D')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date').sort_index()
        df.index = df.index.to_period(freq)
        
        # Select target variable (should be 'value' for stock data)
        target = 'value'
        
        run_button = st.sidebar.button("Run Forecast")
        
        st.subheader(f"Stock Data for {ticker}")
        st.write(df.head())
        
        y = df[target]
        y_train, y_test = manual_train_test_split(y, train_size)
        
        results = {}
        
        if run_button:
            for model in model_choices:
                with st.spinner(f"Running {model}..."):
                    st.write(f"### Running {model}...")
                    params = {}
                    
                    # Set model-specific parameters
                    if model == 'Naive':
                        params['strategy'] = naive_strategy
                        params['window_length'] = naive_window
                    elif model == 'ETS':
                        params['error'] = ets_error
                        params['trend'] = ets_trend
                        params['seasonal'] = ets_seasonal
                        params['sp'] = ets_seasonal_periods  # Use 'sp' instead of 'seasonal_periods'
                    elif model == 'ARIMA':
                        params['p'] = arima_p
                        params['d'] = arima_d
                        params['q'] = arima_q
                    elif model == 'RandomForest':
                        params['n_estimators'] = rf_n_estimators
                        params['max_depth'] = rf_max_depth
                        params['n_lags'] = rf_n_lags
                    elif model == 'RNN':
                        params['units'] = rnn_units
                        params['learning_rate'] = rnn_learning_rate
                        params['epochs'] = rnn_epochs
                        params['n_lags'] = rnn_n_lags
                    elif model == 'LSTM':
                        params['units'] = lstm_units
                        params['learning_rate'] = lstm_learning_rate
                        params['epochs'] = lstm_epochs
                        params['n_lags'] = lstm_n_lags
                    elif model == 'XGBoost':
                        params['n_estimators'] = xgb_n_estimators
                        params['max_depth'] = xgb_max_depth
                        params['learning_rate'] = xgb_learning_rate
                        params['n_lags'] = xgb_n_lags
                        
                    try:
                        forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model, fh, **params)
                        results[model] = {"y_pred": y_pred, "y_forecast": y_forecast}
                    except Exception as e:
                        st.error(f"Error in model {model}: {e}")
            
            if results:
                st.subheader("Forecast Comparison")
                fig = plot_time_series(y_train, y_test, results, "Forecast Results")
                st.pyplot(fig)
    else:
        if data_source == "Yahoo Finance":
            st.info("Please enter a stock ticker and click 'Fetch Stock Data' to run forecasts.")
        else:
            st.info("Please upload a CSV file to run forecasts.")

if __name__ == "__main__":
    main()

