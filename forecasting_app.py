# Author: Prof. Pedram Jahangiry
# Date: 2024-10-10


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    preprocess_data,
    manual_train_test_split,
    calculate_metrics,
    rank_and_style_metrics,
    MODEL_REGISTRY
)


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
        """
    return ""


def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    col1, col2, col3 = st.columns([3, 3, 4])

    with col1:
        st.header("Model Assumptions")
        model_choices = st.multiselect("Select model(s) to run", list(MODEL_REGISTRY.keys()))
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100
        
        model_configs = {}
        for model in model_choices:
            model_params = {}
            with st.expander(f"{model} Parameters", expanded=False):
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
                    seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=1)
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
                    n_lags = st.number_input("Number of lags", min_value=1, value=5)
                    model_params = {"n_lags": n_lags}
            model_configs[model] = model_params       

    with col2:
        st.header("Data Handling")
        use_example = st.checkbox("Use example data", value=False)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if use_example:
            file = "AEP_hourly.csv"
        else:
            file = uploaded_file
            
        if file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(file)

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
                
                    st.subheader("Data Preview")
                    st.write(df_filtered.head())                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
                st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")

    with col3:
        st.header("Forecast Results")
        fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
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
                                status.update(label=f"{model} model completed.", state="complete")
                            except Exception as e:
                                status.update(label=f"Error in {model} model: {str(e)}", state="error")
                                st.error(f"An error occurred while running the {model} model: {str(e)}")
                        
                    # Plot all results
                    st.subheader("Forecast Comparison")
                    st.subheader("Model Comparison Metrics")
                    metrics_df = pd.DataFrame({
                        model: res["metrics"]
                        for model, res in results.items()
                    }).T[["MAE", "RMSE", "MAPE"]]
                    styled_metrics = rank_and_style_metrics(metrics_df, sort_by="MAPE")
                    st.dataframe(styled_metrics)
                    fig = plot_time_series(y_train, y_test, results, f"Forecast Comparison for {target_variable}")
                    st.pyplot(fig)

                    # st.subheader("Test Set Predictions")
                    # st.write(y_pred)

                    # st.subheader("Future Forecast Values")
                    # st.write(y_forecast)
                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable before running the forecast.")

if __name__ == "__main__":
    main()