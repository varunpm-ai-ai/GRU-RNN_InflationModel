import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_gru_model(input_shape):
    """Recreates the GRU model architecture from the notebook."""
    Sequential = tf.keras.models.Sequential
    GRU = tf.keras.layers.GRU
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    l2 = tf.keras.regularizers.l2

    model = Sequential([
        GRU(16, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(8, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1)
    ])
    return model

try:
    model = tf.keras.models.load_model('models/gru_model.h5')
except:
    st.warning("Model file 'gru_model.h5' not found. Using a dummy model structure. **Prediction values will be randomized.** Please check file path.")
    model = create_gru_model((4, 12))
    
N_STEPS = 4
FEATURE_COLUMNS = [
    'CPI', 'GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
    'Gold_Prices', 'Real_Estate_Index', 'Stock_Index', 
    'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
    'Retail_Sales', 'Exchange_Rate'
]

SCALER_X_MEAN = np.array([5.11, 6.31, 6.78, 68.74, 1500.56, 121.36, 39688.35, 9.94, 12.35, 4.67, 7.82, 74.56])
SCALER_X_STD = np.array([0.56, 0.54, 0.40, 6.64, 140.01, 3.49, 1729.87, 1.48, 1.70, 0.59, 0.81, 3.51])
SCALER_Y_MEAN = 4.8858
SCALER_Y_STD = 0.5284

HISTORICAL_DATA_VALUES = [
    [5.756273, 6.343160, 6.971051, 75.890515, 1619.488071, 125.485362, 39307.487673, 10.015411, 13.375982, 5.662490, 8.069475, 72.615050],
    [5.323190, 7.013432, 6.887029, 71.948621, 1368.812355, 117.014727, 40032.619870, 8.705226, 11.538070, 5.123542, 8.472919, 75.334917],
    [4.764910, 5.581241, 7.151573, 72.091241, 1489.798886, 117.520991, 41961.288018, 9.147866, 10.807228, 5.322838, 7.549882, 73.184851]
]
HISTORICAL_DF = pd.DataFrame(HISTORICAL_DATA_VALUES, columns=FEATURE_COLUMNS, index=[2020, 2021, 2022])

TEST_SET_RESULTS = pd.DataFrame({
    'Year': [2021, 2022, 2023],
    'Actual CPI (%)': [5.3232, 4.7649, 4.5710],
    'Predicted CPI (%)': [5.1538, 4.9648, 4.8591] 
})


def single_prediction_step(sequence, model, mean_X, std_X, mean_y, std_y):
    """Scales a 4-step sequence, predicts the next CPI (scalar), and inverse transforms it."""
    
    if sequence.shape != (N_STEPS, len(FEATURE_COLUMNS)):
        raise ValueError("Sequence shape error in single_prediction_step.")
    
    X_scaled = (sequence - mean_X) / std_X

    X_input = X_scaled.reshape(1, N_STEPS, X_scaled.shape[1])
    
    y_pred_s = model.predict(X_input, verbose=0)[0, 0]
    
    y_pred = (y_pred_s * std_y) + mean_y
    
    return y_pred

def multi_step_forecast(latest_data_2023, historical_df, model, target_year):
    
    forecast_results = {}
    current_year = 2024
    
    latest_sequence_data = historical_df.tail(N_STEPS - 1).values
    
    current_sequence = np.vstack([latest_sequence_data, np.array(list(latest_data_2023.values()))])

    while current_year <= target_year:
        
        predicted_cpi = single_prediction_step(
            current_sequence, 
            model, 
            SCALER_X_MEAN, SCALER_X_STD, 
            SCALER_Y_MEAN, SCALER_Y_STD
        )
        
        forecast_results[current_year] = predicted_cpi
        
        if current_year == target_year:
            break
            
        next_sequence = current_sequence[1:].copy()
        
        new_time_step = np.array([
            predicted_cpi, 
            latest_data_2023['GDP_Growth'], 
            latest_data_2023['Unemployment_Rate'], 
            latest_data_2023['Crude_Oil_Prices'], 
            latest_data_2023['Gold_Prices'], 
            latest_data_2023['Real_Estate_Index'], 
            latest_data_2023['Stock_Index'], 
            latest_data_2023['Money_Supply_M1'], 
            latest_data_2023['Money_Supply_M2'], 
            latest_data_2023['Industrial_Production'], 
            latest_data_2023['Retail_Sales'], 
            latest_data_2023['Exchange_Rate']
        ])
        
        current_sequence = np.vstack([next_sequence, new_time_step.reshape(1, -1)])
        
        current_year += 1

    return forecast_results


st.set_page_config(
    page_title="India Inflation (CPI) Multi-Year Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main-header { font-weight: 800; color: #1F2937; font-size: 2.5rem; }
    .stButton>button { background-color: #2563EB; color: white; font-weight: 600; border-radius: 0.5rem; padding: 0.75rem 1.5rem; border: none; transition: all 0.2s; }
    .stButton>button:hover { background-color: #1D4ED8; transform: translateY(-2px); }
    .stNumberInput, .stTextInput { border: 1px solid #D1D5DB; border-radius: 0.375rem; padding: 0.5rem; }
    .stAlert { border-radius: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">📈 India CPI Multi-Year Forecast (GRU Model)</p>', unsafe_allow_html=True)

st.write("""
This application uses your trained GRU model to predict the Consumer Price Index (CPI) year-by-year. 
For forecasts beyond 2024, the model uses the *predicted CPI* from the previous year, 
while holding the other 11 economic factors constant at your 2023 input values.
""")

latest_inputs = {}
input_labels = {
    'CPI': 'Current Year CPI (2023 Average)',
    'GDP_Growth': 'GDP Growth Rate (%)',
    'Unemployment_Rate': 'Unemployment Rate (%)',
    'Crude_Oil_Prices': 'Crude Oil Prices ($/barrel)',
    'Gold_Prices': 'Gold Prices ($/ounce)',
    'Real_Estate_Index': 'Real Estate Index',
    'Stock_Index': 'Stock Index (e.g., Nifty/Sensex)',
    'Money_Supply_M1': 'Money Supply M1 Growth (%)',
    'Money_Supply_M2': 'Money Supply M2 Growth (%)',
    'Industrial_Production': 'Industrial Production Growth (%)',
    'Retail_Sales': 'Retail Sales Growth (%)',
    'Exchange_Rate': 'Exchange Rate (Local/USD)'
}

RANGE_CONSTRAINTS = {
    'CPI': {'min': 1.0, 'max': 10.0},
    'GDP_Growth': {'min': -5.0, 'max': 15.0},
    'Unemployment_Rate': {'min': 3.0, 'max': 12.0},
    'Crude_Oil_Prices': {'min': 30.0, 'max': 120.0},
    'Gold_Prices': {'min': 1000.0, 'max': 2500.0},
    'Real_Estate_Index': {'min': 80.0, 'max': 160.0},
    'Stock_Index': {'min': 25000.0, 'max': 65000.0}, 
    'Money_Supply_M1': {'min': 1.0, 'max': 20.0},
    'Money_Supply_M2': {'min': 3.0, 'max': 22.0},
    'Industrial_Production': {'min': -2.0, 'max': 12.0},
    'Retail_Sales': {'min': 1.0, 'max': 14.0},
    'Exchange_Rate': {'min': 60.0, 'max': 95.0}
}


with st.form("inflation_form"):
    st.markdown("##### 1. Define Forecast Horizon")
    target_year = st.number_input(
        "Predict CPI up to the end of year:",
        min_value=2024,
        max_value=2035,
        value=2028,
        step=1
    )
    st.markdown("##### 2. Enter Estimated Economic Indicators for 2023 (Annual Averages)")
    st.info("These 2023 inputs will be used as the base case for all subsequent forecast years.")

    cols = st.columns(3)
    
    for i, (key, label) in enumerate(input_labels.items()):
        col = cols[i % 3]
        with col:
            default_value = HISTORICAL_DF.iloc[-1][key] if key != 'CPI' else 4.5710
            
            latest_inputs[key] = st.number_input(
                label=label,
                min_value=RANGE_CONSTRAINTS.get(key, {}).get('min', 0.0), 
                max_value=RANGE_CONSTRAINTS.get(key, {}).get('max', 100.0), 
                value=float(f"{default_value:.4f}"),
                format="%.4f"
            )

    submitted = st.form_submit_button(f"Generate Forecast up to {target_year}", type="primary")

if submitted:
    try:
        latest_data_2023 = {k: v for k, v in latest_inputs.items()}
        ordered_latest_inputs = {col: latest_inputs[col] for col in FEATURE_COLUMNS}
        forecast_results = multi_step_forecast(
            ordered_latest_inputs, 
            HISTORICAL_DF, 
            model, 
            target_year
        )
        historical_df = TEST_SET_RESULTS.copy()
        historical_df['Type'] = 'Actual (Validation)'
        historical_df.loc[historical_df['Year'] == 2023, 'Type'] = 'Latest Input (2023 Actual)'
        historical_df.rename(columns={'Actual CPI (%)': 'CPI (%)'}, inplace=True)
        
        forecast_list = [{'Year': year, 'CPI (%)': cpi} for year, cpi in forecast_results.items()]
        forecast_df = pd.DataFrame(forecast_list)
        forecast_df['Type'] = 'Forecast'
        
        plot_data = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        st.markdown("### Multi-Year CPI Forecast Results", unsafe_allow_html=True)
        
        st.line_chart(
            plot_data,
            x='Year',
            y='CPI (%)',
            color='Type',
            height=350
        )

        st.markdown(f"##### Predicted CPI Values (2024 - {target_year})")
        st.dataframe(
            forecast_df[['Year', 'CPI (%)']].set_index('Year').style.format("{:.4f}"),
            use_container_width=True
        )

        st.success(f"Forecast successfully generated up to {target_year}. Note the stabilization/trend based on your initial 2023 economic inputs.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the model file is correctly loaded and the input format is consistent.")
        st.exception(e)
        
st.markdown("---")
st.markdown("#### Model Context and Test Performance")
st.write("The GRU model uses a 4-year sequence to make a 1-year ahead prediction. The initial sequence is built from the latest actual data available (2020-2022) and your 2023 inputs.")

st.markdown("##### Model's Validation on Test Data (2021-2023)")
st.dataframe(TEST_SET_RESULTS.set_index('Year').style.format({
    'CPI_actual': "{:.4f}",
    'CPI_pred': "{:.4f}"
}), use_container_width=True)

st.markdown("##### Last 3 Historical Data Points (2020-2022 Averages)")
st.dataframe(HISTORICAL_DF.style.format("{:.4f}"))