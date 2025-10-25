import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

N_STEPS = 4
FEATURE_COLUMNS = [
    'CPI', 'GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
    'Gold_Prices', 'Real_Estate_Index', 'Stock_Index', 
    'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
    'Retail_Sales', 'Exchange_Rate'
]
PREDICTION_COLUMNS = ['CPI (%)', 'GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
                      'Gold_Prices', 'Real_Estate_Index', 'Stock_Index', 
                      'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
                      'Retail_Sales', 'Exchange_Rate']

SCALER_X_MEAN = np.array([5.11, 6.31, 6.78, 68.74, 1500.56, 121.36, 39688.35, 9.94, 12.35, 4.67, 7.82, 74.56])
SCALER_X_STD = np.array([0.56, 0.54, 0.40, 6.64, 140.01, 3.49, 1729.87, 1.48, 1.70, 0.59, 0.81, 3.51])
SCALER_Y_MEAN = 4.8858
SCALER_Y_STD = 0.5284

HISTORICAL_DATA_VALUES = [
    [4.501134, 5.358729, 6.547631, 63.265379, 1576.8483, 120.7920, 39388.8496, 10.7778, 12.6093, 4.2982, 7.4838, 74.9727],
    [5.071874, 6.792688, 6.462077, 75.9269, 1479.5634, 124.3498, 39821.4412, 12.0038, 13.5465, 5.7121, 7.5854, 74.0937],
    [5.218608, 4.857653, 6.886933, 65.6989, 1555.7316, 115.0315, 40080.8078, 9.8715, 11.7553, 4.7834, 8.4435, 75.2893],
    [4.840034, 6.176862, 7.384744, 69.3837, 1551.3352, 120.8998, 41744.8148, 10.0986, 12.7711, 5.1118, 7.5740, 73.1893],
    [4.762224, 5.916498, 7.141215, 70.8097, 1667.4531, 123.6854, 38748.0183, 10.7225, 12.3680, 4.8214, 7.9410, 76.2672],
    [4.956278, 6.566733, 6.5920, 67.1892, 1521.2606, 118.7768, 40111.7679, 11.3301, 13.0852, 5.1923, 7.7973, 77.6736],
    [5.195231, 6.792853, 6.8844, 70.6159, 1587.8806, 128.4946, 40347.1023, 10.1042, 11.6812, 4.9432, 7.1429, 76.3647],
    [4.902143, 6.339809, 6.2967, 65.3063, 1529.9308, 123.5395, 40061.5321, 10.3814, 12.9158, 4.7667, 8.8850, 76.5476],
    [4.828861, 5.681420, 6.8980, 70.5085, 1430.0194, 126.3766, 40727.6818, 9.2506, 12.0898, 5.3863, 7.9857, 75.2390],
    [4.878889, 5.5727, 6.8151, 65.8637, 1501.5844, 114.4984, 36422.6594, 9.8238, 11.8382, 5.0638, 7.4919, 75.1548],
    [5.756273, 6.3432, 6.9711, 75.8905, 1619.4881, 125.4854, 39307.4877, 10.0154, 13.3760, 5.6625, 8.0695, 72.6151],
    [5.323190, 7.0134, 6.8870, 71.9486, 1368.8124, 117.0147, 40032.6199, 8.7052, 11.5381, 5.1235, 8.4729, 75.3349],
    [4.764910, 5.5812, 7.1516, 72.0912, 1489.7989, 117.5210, 41961.2880, 9.1479, 10.8072, 5.3228, 7.5499, 73.1849]
]
HISTORICAL_DF = pd.DataFrame(HISTORICAL_DATA_VALUES, columns=FEATURE_COLUMNS, index=range(2010, 2023))

FULL_HISTORICAL_CPI_DATA = pd.DataFrame({
    'Year': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'CPI (%)': [5.4442, 4.1132, 4.7100, 4.5011, 5.0719, 5.2186, 4.8400, 4.7622, 4.9563, 5.1952, 4.9021, 4.8289, 4.8789, 5.7563, 5.3232, 4.7649, 4.5710],
})
FULL_HISTORICAL_CPI_DATA['Type'] = 'Historical CPI (Actual)'
FULL_HISTORICAL_CPI_DATA.loc[FULL_HISTORICAL_CPI_DATA['Year'] == 2023, 'Type'] = 'Latest Input (2023 Actual)'


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
    model_path = os.path.join(os.path.dirname(__file__), 'gru_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    st.warning(f"Model file 'gru_model.h5' not found at {model_path}. Using a dummy model structure. **Prediction values will be randomized.** Please check file path.")
    model = create_gru_model((N_STEPS, len(FEATURE_COLUMNS)))
    
def single_prediction_step(sequence, model, mean_X, std_X, mean_y, std_y):
    """Scales a 4-step sequence, predicts the next CPI (scalar), and inverse transforms it."""
    
    if sequence.shape != (N_STEPS, len(FEATURE_COLUMNS)):
        if model.name == 'sequential': 
            return (np.random.rand() * 4) + 2 
        else:
            raise ValueError("Sequence shape error in single_prediction_step.")
    
    X_scaled = (sequence - mean_X) / std_X

    X_input = X_scaled.reshape(1, N_STEPS, X_scaled.shape[1])
    
    y_pred_s = model.predict(X_input, verbose=0)[0, 0]
    
    y_pred = (y_pred_s * std_y) + mean_y
    
    return y_pred

def multi_step_forecast(latest_data_2023, historical_df, model, target_year):
    
    full_forecast_data = []
    current_year = 2024  
    latest_sequence_data = historical_df.tail(N_STEPS - 1).values
    latest_data_row = np.array([latest_data_2023[col] for col in FEATURE_COLUMNS])
    current_sequence = np.vstack([latest_sequence_data, latest_data_row.reshape(1, -1)])
    if current_sequence.shape[0] != N_STEPS or current_sequence.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Initial sequence formation failed. Expected ({N_STEPS}, {len(FEATURE_COLUMNS)}), got {current_sequence.shape}")
    
    while current_year <= target_year:
        predicted_cpi = single_prediction_step(
            current_sequence, 
            model, 
            SCALER_X_MEAN, SCALER_X_STD, 
            SCALER_Y_MEAN, SCALER_Y_STD
        )
        
        forecast_row = {
            'Year': current_year,
            'CPI (%)': predicted_cpi,
            'GDP_Growth': latest_data_2023['GDP_Growth'],
            'Unemployment_Rate': latest_data_2023['Unemployment_Rate'],
            'Crude_Oil_Prices': latest_data_2023['Crude_Oil_Prices'],
            'Gold_Prices': latest_data_2023['Gold_Prices'],
            'Real_Estate_Index': latest_data_2023['Real_Estate_Index'],
            'Stock_Index': latest_data_2023['Stock_Index'],
            'Money_Supply_M1': latest_data_2023['Money_Supply_M1'],
            'Money_Supply_M2': latest_data_2023['Money_Supply_M2'],
            'Industrial_Production': latest_data_2023['Industrial_Production'],
            'Retail_Sales': latest_data_2023['Retail_Sales'],
            'Exchange_Rate': latest_data_2023['Exchange_Rate']
        }
        
        full_forecast_data.append(forecast_row)
        
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
    return pd.DataFrame(full_forecast_data)

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
For forecasts beyond 2024, the model uses the **predicted CPI** from the previous year, 
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
    st.info("These **2023 inputs** will be used as the **constant base case** for all subsequent forecast years for all factors *except* the predicted CPI.")

    cols = st.columns(3)
    
    for i, (key, label) in enumerate(input_labels.items()):
        col = cols[i % 3]
        with col:
            default_value = FULL_HISTORICAL_CPI_DATA.loc[FULL_HISTORICAL_CPI_DATA['Year'] == 2023, 'CPI (%)'].iloc[0] if key == 'CPI' else HISTORICAL_DF.iloc[-1][key]
            
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
        forecast_df = multi_step_forecast(
            ordered_latest_inputs, 
            HISTORICAL_DF, 
            model, 
            target_year
        )
        historical_plot_df = FULL_HISTORICAL_CPI_DATA.copy()
        historical_plot_df.loc[historical_plot_df['Year'] == 2023, 'CPI (%)'] = latest_data_2023['CPI']
        
        actual_data_df = historical_plot_df[historical_plot_df['Year'] <= 2023].copy()
        actual_data_df['Type'] = 'Historical CPI (Actual)'
        actual_data_df.loc[actual_data_df['Year'] == 2023, 'Type'] = 'Latest Input (2023 Actual)'
        
        plot_forecast_df = forecast_df[['Year', 'CPI (%)']].copy()
        plot_forecast_df['Type'] = 'Forecast (Predicted)'
        plot_data = pd.concat([actual_data_df, plot_forecast_df], ignore_index=True)
        
        st.markdown("### Multi-Year CPI Forecast Results", unsafe_allow_html=True)
        st.line_chart(
            plot_data,
            x='Year',
            y='CPI (%)',
            color='Type', 
            height=350
        )
        st.markdown(f"##### Full Predicted Economic Indicators ({HISTORICAL_DF.index.max() + 1} - {target_year})")
        st.write("CPI is predicted iteratively; all other factors are held constant at your 2023 input.")
        st.dataframe(
            forecast_df.set_index('Year').style.format("{:.4f}"),
            use_container_width=True
        )
        st.success(f"Forecast successfully generated up to {target_year}.")
        st.markdown("---")
        st.markdown("#### Model Context and Latest Data")
        st.write("The chart above shows the full historical CPI trend (2007-2023) combined with the multi-year forecast (2024+).")
        st.markdown("##### Latest Historical Data Used to Start Forecast (2010-2022 Averages)")
        st.dataframe(HISTORICAL_DF.style.format("{:.4f}"), use_container_width=True)
        st.markdown("##### Your 2023 Input Values (Used as base case for forecast)")
        st.dataframe(pd.DataFrame(latest_inputs, index=['2023 Input']).T.style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the model file is correctly loaded and the input format is consistent.")
        st.exception(e)