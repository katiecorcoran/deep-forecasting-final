import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    preprocess_data,
    manual_train_test_split,
    calculate_metrics,
    rank_and_style_metrics,
    fetch_stock_data,
    MODEL_REGISTRY
)
from datetime import datetime, timedelta


def plot_time_series(y_train, y_test, results, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    for model, result in results.items():
        y_pred = result['y_pred']
        y_forecast = result['y_forecast']
        # y_pred may be empty for a multioutput strategy
        if not y_pred.empty:
            ax.plot(y_pred.index.to_timestamp(), y_pred.values, label=f"{model} Test Predictions")
        ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label=f"{model} Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig

def get_model_help_text(model):
    if model == "Naive":
        return """
        **Naive Parameters:**
        - `strategy`: Method for forecasting:
            - `'last'`: repeat last value
            - `'mean'`: average over a window
            - `'drift'`: linear extrapolation
        - `window_length`: Number of past values used for `'mean'`
        """
    elif model == "ETS":
        return """
        **ETS Parameters:**
        - `error`: Type of error term (`add`, `mul`)
        - `trend`: Trend type (`add`, `mul`, or `None`)
        - `seasonal`: Seasonal component (`add`, `mul`, or `None`)
        - `damped_trend`: If the trend should flatten over time
        - `sp`: Seasonal period
        """
    elif model == "ARIMA":
        return """
        **ARIMA Parameters:**
        - `p`, `d`, `q`: Non-seasonal ARIMA components
        - `P`, `D`, `Q`, `sp`: Seasonal ARIMA components
        - `seasonal`: Whether to include seasonality
        """
    elif model == "RandomForest":
        return """
        **Random Forest Parameters:**
        - `n_lags`: Number of previous observations used as features
        - `difference_data`: If True, the data is differenced before training
        """
    elif model == "NeuralNet":
        return """
        **NeuralNet Parameters:**
        
        - `n_lags`: Number of previous time steps used as input features.
        - `units`: Number of RNN units (neurons) in the layer.
            * This controls the number of nodes/neurons in the RNN, LSTM, and XGBoost models.
                * For small datasets (<100 points): 5-10 nodes
                * For medium datasets (100-1000 points): 10-20 nodes
                * For large datasets (>1000 points): 20-50 nodes
                                     Start with fewer nodes and increase if needed.
        - `n_epochs`: Number of training epochs.
        - `batch_size`: Number of samples per training batch.
        - `learning_rate`: Learning rate for the optimizer.
        - `model_type`: Type of RNN model to use (e.g., RNN, LSTM).
        It is recommended to refresh between Neural Net models for optimal performance.
        """
    elif model == "XGBoost":
        return """
        **XGBoost Parameters:**

        - `n_estimators`: Number of boosting rounds (trees).
        - `max_depth`: Maximum tree depth. Higher = more complex trees.
        - `learning_rate`: Step size shrinkage used to prevent overfitting.
        - `n_lags`: Number of previous time steps used as input features.
        """
    return ""


def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    df = st.session_state.get("df", None)

    st.sidebar.subheader("Model Assumptions")
    model_choices = st.sidebar.multiselect("Select model(s) to run", list(MODEL_REGISTRY.keys()))
    train_size = st.sidebar.slider("Train size (%)", 50, 95, 80) / 100
    model_configs = {}
    for model in model_choices:
        model_params = {}
        with st.sidebar.expander(f"{model} Parameters", expanded=False):
            if model == "Naive":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                strategy = st.selectbox("Naive strategy", ["last", "mean", "drift"])
                window_length = None
                if strategy == "mean":
                    window_length = st.number_input("Window length (optional)", min_value=1, value=5)
                model_params = {
                    "strategy": strategy,
                    "window_length": window_length if window_length else None,
                }
            elif model == "ETS":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                error = st.selectbox("Error type", ["add", "mul"])
                trend = st.selectbox("Trend type", ["add", "mul", None])
                seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
                damped_trend = st.checkbox("Damped trend", value=False)
                seasonal_periods = st.number_input("Seasonal periods", min_value=2, value=12, 
                                                    help="Must be greater than 1. For monthly data use 12, quarterly use 4, etc.")
                model_params = {
                    "error": error,
                    "trend": trend,
                    "seasonal": seasonal,
                    "damped_trend": damped_trend,
                    "sp": seasonal_periods,
                }
            elif model == "ARIMA":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                st.subheader("Non-seasonal")
                start_p = st.number_input("Min p", min_value=0, value=0)
                max_p = st.number_input("Max p", min_value=0, value=5)
                start_q = st.number_input("Min q", min_value=0, value=0)
                max_q = st.number_input("Max q", min_value=0, value=5)
                d = st.number_input("d", min_value=0, value=1)
                
                st.subheader("Seasonal")
                seasonal = st.checkbox("Seasonal", value=True)
                if seasonal:
                    start_P = st.number_input("Min P", min_value=0, value=0)
                    max_P = st.number_input("Max P", min_value=0, value=2)
                    start_Q = st.number_input("Min Q", min_value=0, value=0)
                    max_Q = st.number_input("Max Q", min_value=0, value=2)
                    D = st.number_input("D", min_value=0, value=1)
                    sp = st.number_input("Periods", min_value=1, value=12)
                
                model_params = {
                    "start_p": start_p,
                    "max_p": max_p,
                    "start_q": start_q,
                    "max_q": max_q,
                    "d": d,
                    "seasonal": seasonal,
                }
                if seasonal:
                    model_params.update({
                        "start_P": start_P,
                        "max_P": max_P,
                        "start_Q": start_Q,
                        "max_Q": max_Q,
                        "D": D,
                        "sp": sp
                    })
            elif model == "RandomForest":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                n_lags = st.number_input("RF Number of lags", min_value=1, value=5)
                difference_data = st.checkbox("Difference data", value=False)
                model_params = {"n_lags": n_lags, "difference_data": difference_data}
            elif model == "NeuralNet":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                n_lags = st.number_input("RNN Number of lags", min_value=1, value=5)
                units = st.number_input("Number of RNN units", min_value=1, value=50)
                n_epochs = st.number_input("Number of epochs", min_value=1, value=10)
                batch_size = st.number_input("Batch size", min_value=1, value=32)
                learning_rate = st.number_input("Learning rate", min_value=0.0001, value=0.001, format="%.4f")
                model_type = st.selectbox("Model type", ["RNN", "LSTM"])
                model_params = {"n_lags": n_lags, "n_epochs": n_epochs, "batch_size": batch_size, "units": units, "learning_rate": learning_rate, "model_type": model_type}
            elif model == "XGBoost":
                if st.checkbox(f"Show help for {model} parameters", key=f"{model}_help"):
                    st.info(get_model_help_text(model))
                xgb_n_estimators = st.number_input("XGBoost Number of Trees", min_value=10, value=100)
                xgb_max_depth = st.number_input("XGBoost Max Depth", min_value=1, value=5)
                xgb_learning_rate = st.number_input("XGBoost Learning Rate", min_value=0.01, value=0.1, format="%.2f")
                xgb_n_lags = st.number_input("XGBoost Number of Lags", min_value=1, value=10)
                model_params = {
                    "n_estimators": xgb_n_estimators,
                    "max_depth": xgb_max_depth,
                    "learning_rate": xgb_learning_rate,
                    "n_lags": xgb_n_lags,
                }
        model_configs[model] = model_params       
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("Data Source")
        data_source = st.radio("Choose data source", ["Upload CSV", "Use Example Data", "Yahoo Finance"])
        if data_source == "Upload CSV":
            st.info("Upload your own CSV file.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully.")
        elif data_source == "Use Example Data":
            st.info("Using example data: AEP_hourly.csv")
            df = pd.read_csv("AEP_hourly.csv")
            st.success("Example data loaded successfully.")
        elif data_source == "Yahoo Finance":
            st.info("Using Yahoo Finance data. Please enter the stock ticker and date range.")
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
                        # Convert frequency string to pandas frequency
                        freq_map = {'1d': 'D', '1wk': 'W', '1mo': 'M'}
                        pandas_freq = freq_map[freq]
                        
                        # Ensure the index has the correct frequency
                        df.index = pd.DatetimeIndex(df.index).to_period(pandas_freq)
                        
                        st.success(f"Successfully fetched data for {ticker}")
                    else:
                        st.error(f"Failed to fetch data for {ticker}")
            
        if df is not None:
            st.session_state.df = df
            try:
                # Allow user to select the frequency
                freq_options = ['D', 'W', 'M', 'Q', 'Y', 'H']
                freq = st.selectbox("Select the data frequency", freq_options)
                numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()

                if not numeric_columns:
                    st.error("No numeric columns found in the uploaded data. Please ensure your CSV contains numeric data for forecasting.")
                else:
                    df = preprocess_data(df, freq)
                    min_date = df.index.min().to_timestamp().date()
                    max_date = df.index.max().to_timestamp().date()
                    start_date = st.date_input(
                        "Select start date for analysis",
                        min_value=min_date,
                        max_value=max_date,
                        value=min_date
                    )
                    df_filtered = df.loc[df.index.to_timestamp().date >= start_date]
                    
                    target_variable = st.selectbox("Select your target variable", numeric_columns)
                    y = df_filtered[target_variable]
                    
                    st.subheader(f"Time Series Plot: {target_variable}")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(y.index.to_timestamp(), y.values)
                    st.pyplot(fig)
                    
                    difference_plot = st.toggle("Show Differenced Plot", value=False)
                    if difference_plot:
                        st.subheader("Differenced Time Series Plot")
                        y_diff = y.diff().dropna()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(y_diff.index.to_timestamp(), y_diff.values)
                        st.pyplot(fig)
                
                    st.subheader("Data Preview")
                    st.write(df_filtered.head())                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
                st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")
    
    with col2:
        st.subheader("Forecasting Results")
        
        # Create a container for the parameters section
        with st.container():            
            st.markdown("""
            **Forecast Horizon**  
            Number of future periods to predict.  
            *Higher values predict further into the future (e.g., more days, months, or years ahead) but may be less accurate.*
            """)
            fh = st.number_input("Number of periods to forecast", 
                                min_value=1, 
                                value=10,
                                help="Choose based on your planning horizon (e.g., 12 for yearly data, 30 for daily data)")
        
        # Add a separator
        st.markdown("---")
        
        run_forecast_button = st.button("Run Forecast")
        
        if run_forecast_button:
            if 'y' in locals():
                try:
                    # Perform train-test split
                    y_train, y_test = manual_train_test_split(y, train_size)
                    results = {}
                    for model in model_choices:
                        with st.status(f"Running {model} model...") as status:
                            try:
                                model_params = model_configs.get(model, {})
                                forecaster_func = MODEL_REGISTRY[model]
                                forecaster, y_pred, y_forecast = forecaster_func(y_train, y_test, fh, **model_params)
                                metrics = calculate_metrics(y_test.loc[y_pred.index], y_pred)
                                results[model] = {
                                    "forecaster": forecaster,
                                    "y_pred": y_pred,
                                    "y_forecast": y_forecast,
                                    "metrics": metrics,
                                }
                                st.write("### Forecasted Values")
                                st.dataframe(y_forecast)
                                st.write("### Model Metrics")
                                st.dataframe(metrics)
                                status.update(label=f"{model} model completed.", state="complete")
                            except Exception as e:
                                status.update(label=f"Error in {model} model: {str(e)}", state="error")
                                st.error(f"An error occurred while running the {model} model: {str(e)}")
                        
                    # Plot all results
                    st.subheader("Forecast Comparison")
                    
                    metrics_df = pd.DataFrame({
                        model: res["metrics"]
                        for model, res in results.items()
                    }).T[["MAE", "RMSE", "MAPE"]]
                    styled_metrics = rank_and_style_metrics(metrics_df, sort_by="MAPE")
                    st.dataframe(styled_metrics)
                    
                    fig = plot_time_series(y_train, y_test, results, f"Forecast Comparison for {target_variable}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable before running the forecast.")

if __name__ == "__main__":
    main()