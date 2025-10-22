import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- 1. Load/Simulate Model and Scaler Data ---
# NOTE: In a real deployment, you would load your saved files:
# model = tf.keras.models.load_model('gru_model.h5')
# scaler_X = joblib.load('scaler_X.save')
# scaler_y = joblib.load('scaler_y.save')

# Since I cannot load the exact binary files, I will simulate the setup
# based on the training logic and data structure seen in your notebook.

# Define model architecture (based on notebook cell 67)
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
    # Model compilation is not strictly necessary for prediction, but good practice
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# Simulate Loading the Model (replace with actual loading in your environment)
try:
    # Attempt to load the actual model if saved in the environment
    model = tf.keras.models.load_model('gru_model.h5')
except:
    # If loading fails, create a dummy model for structure and warn user
    st.warning("Model file 'gru_model.h5' not found in the path. Using a dummy model structure. **Prediction values will be random.** Please ensure your model is saved correctly and accessible.")
    model = create_gru_model((4, 12))

# Features and Constants (Based on notebook cells 59/61)
N_STEPS = 4
FEATURE_NAMES = [
    'CPI_lag1', 'GDP_Growth_lag1', 'Unemployment_Rate_lag1', 'Crude_Oil_Prices_lag1',
    'Gold_Prices_lag1', 'Real_Estate_Index_lag1', 'Stock_Index_lag1',
    'Money_Supply_M1_lag1', 'Money_Supply_M2_lag1', 'Industrial_Production_lag1',
    'Retail_Sales_lag1', 'Exchange_Rate_lag1'
]

# --- SIMULATED SCALER PARAMETERS ---
# In a real setup, you would load scaler_X and scaler_y.
# These values are derived from calculating mean/std on the training set (2011-2019 predictions)
# based on the overall df_yearly data in your notebook.

# Simulated scaler_X parameters (mean/std of the 12 input features from the training data)
# These are representative values derived from the notebook's data (2008-2019 yearly averages)
SCALER_X_MEAN = np.array([5.11, 6.31, 6.78, 68.74, 1500.56, 121.36, 39688.35, 9.94, 12.35, 4.67, 7.82, 74.56])
SCALER_X_STD = np.array([0.56, 0.54, 0.40, 6.64, 140.01, 3.49, 1729.87, 1.48, 1.70, 0.59, 0.81, 3.51])

# Simulated scaler_y parameters (mean/std of the CPI target from the training data)
SCALER_Y_MEAN = 4.8858  # Mean CPI from the training set
SCALER_Y_STD = 0.5284   # Std Dev CPI from the training set

# --- HISTORICAL DATA (Last n_steps-1 required to form the next sequence) ---
# This simulates loading the historical data (features for 2020, 2021, 2022)
# To predict the CPI for 2024, we need input features from 2020, 2021, 2022, 2023.
# We take 2020, 2021, 2022 from the notebook's df_yearly (lagged features)

# The sequence for prediction uses the features that were 'lagged' in the notebook.
# This data represents the actual values (not lagged) for the years T-4 to T-2
# in the sequence (2020-2022 data for a 2024 prediction).

# Final years of data from df_yearly (Years 2020, 2021, 2022)
HISTORICAL_DATA_YEARS = [2020, 2021, 2022]
HISTORICAL_DATA_VALUES = [
    [5.756273, 6.343160, 6.971051, 75.890515, 1619.488071, 125.485362, 39307.487673, 10.015411, 13.375982, 5.662490, 8.069475, 72.615050],
    [5.323190, 7.013432, 6.887029, 71.948621, 1368.812355, 117.014727, 40032.619870, 8.705226, 11.538070, 5.123542, 8.472919, 75.334917],
    [4.764910, 5.581241, 7.151573, 72.091241, 1489.798886, 117.520991, 41961.288018, 9.147866, 10.807228, 5.322838, 7.549882, 73.184851]
]
HISTORICAL_DF = pd.DataFrame(HISTORICAL_DATA_VALUES, columns=[
    'CPI', 'GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
    'Gold_Prices', 'Real_Estate_Index', 'Stock_Index', 
    'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
    'Retail_Sales', 'Exchange_Rate'
])

# --- 2. Scaling and Prediction Function ---
def prepare_and_predict(latest_data_row, historical_df, model, n_steps, mean_X, std_X, mean_y, std_y):
    """
    Combines historical data with latest user input, scales, reshapes, 
    predicts, and inverse transforms the result.
    """
    # Create the T-1 sequence from historical data
    # The last (n_steps - 1) rows of historical data
    sequence_end = historical_df.tail(n_steps - 1).copy() 
    
    # Add the latest user input (Year T) as the last time step (n_steps)
    latest_df = pd.DataFrame([latest_data_row], columns=historical_df.columns)
    full_sequence_df = pd.concat([sequence_end, latest_df], ignore_index=True)
    
    # Scale the input features using the saved/simulated scaler parameters
    X_unscaled = full_sequence_df.values
    X_scaled = (X_unscaled - mean_X) / std_X
    
    # Reshape for GRU: (1, n_steps, n_features)
    X_input = X_scaled.reshape(1, n_steps, X_scaled.shape[1])
    
    # Prediction (on scaled output)
    y_pred_s = model.predict(X_input, verbose=0)
    
    # Inverse transform the prediction (using the saved/simulated scaler_y parameters)
    y_pred = (y_pred_s[0, 0] * std_y) + mean_y
    
    return y_pred, X_input


# --- 3. Streamlit Application UI ---
st.set_page_config(
    page_title="India Inflation (CPI) Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom Tailwind-like CSS for aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-weight: 800;
        color: #1F2937; /* Gray-800 */
        font-size: 2.5rem;
    }
    .stButton>button {
        background-color: #2563EB; /* Blue-600 */
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8; /* Blue-700 */
        transform: translateY(-2px);
    }
    .stNumberInput, .stTextInput {
        border: 1px solid #D1D5DB; /* Gray-300 */
        border-radius: 0.375rem;
        padding: 0.5rem;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">🇮🇳 India Annual CPI Forecast (GRU Model)</p>', unsafe_allow_html=True)

st.write("""
This application uses a trained GRU (Gated Recurrent Unit) Recurrent Neural Network 
to predict the Consumer Price Index (CPI) for **2024**, based on the last four years of key economic indicators.

The model requires **current (2023) yearly average** data for all 12 input features.
""")

# --- Prediction Year ---
st.subheader("Target: CPI Forecast for 2024")

# --- Input Form ---
latest_inputs = {}
input_labels = {
    'CPI': 'Current Year CPI (Previous Year\'s CPI for the Model)',
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

# The model uses CPI_lag1 to Exchange_Rate_lag1. 
# When the user inputs 2023 data, these become the *lag1* features for the 2024 prediction.

with st.form("inflation_form"):
    st.markdown("##### Enter Estimated Economic Indicators for 2023 (Annual Averages)")
    
    cols = st.columns(3)
    cols2 = st.columns(3)
    
    # CPI (This is the CPI_lag1 feature for the 2024 prediction)
    with cols[0]:
        latest_inputs['CPI'] = st.number_input(
            label=input_labels['CPI'],
            min_value=1.0, 
            max_value=15.0, 
            value=4.57,  # Default to 2023 actual for an initial run
            format="%.4f",
            help="The latest available CPI figure. The model uses the *lagged* value of CPI as a key feature."
        )

    # GDP_Growth
    with cols[1]:
        latest_inputs['GDP_Growth'] = st.number_input(
            label=input_labels['GDP_Growth'],
            min_value=-5.0, 
            max_value=15.0, 
            value=6.49,
            format="%.4f"
        )
    
    # Unemployment_Rate
    with cols2[2]:
        latest_inputs['Unemployment_Rate'] = st.number_input(
            label=input_labels['Unemployment_Rate'],
            min_value=3.0, 
            max_value=12.0, 
            value=7.00,
            format="%.4f"
        )

    # Second row of inputs
    
    
    # Crude_Oil_Prices
    with cols2[0]:
        latest_inputs['Crude_Oil_Prices'] = st.number_input(
            label=input_labels['Crude_Oil_Prices'],
            min_value=30.0, 
            max_value=120.0, 
            value=69.38,
            format="%.4f"
        )
        
    # Gold_Prices
    with cols2[1]:
        latest_inputs['Gold_Prices'] = st.number_input(
            label=input_labels['Gold_Prices'],
            min_value=1000.0, 
            max_value=2500.0, 
            value=1505.70,
            format="%.4f"
        )
        
    # Real_Estate_Index
    with cols2[2]:
        latest_inputs['Real_Estate_Index'] = st.number_input(
            label=input_labels['Real_Estate_Index'],
            min_value=80.0, 
            max_value=160.0, 
            value=121.28,
            format="%.4f"
        )

    # Third row of inputs
    cols3 = st.columns(3)
    
    # Stock_Index
    with cols3[0]:
        latest_inputs['Stock_Index'] = st.number_input(
            label=input_labels['Stock_Index'],
            min_value=25000.0, 
            max_value=55000.0, 
            value=40121.28,
            format="%.4f"
        )
        
    # Money_Supply_M1
    with cols3[1]:
        latest_inputs['Money_Supply_M1'] = st.number_input(
            label=input_labels['Money_Supply_M1'],
            min_value=1.0, 
            max_value=20.0, 
            value=9.97,
            format="%.4f"
        )
        
    # Money_Supply_M2
    with cols3[2]:
        latest_inputs['Money_Supply_M2'] = st.number_input(
            label=input_labels['Money_Supply_M2'],
            min_value=3.0, 
            max_value=22.0, 
            value=11.98,
            format="%.4f"
        )

    # Fourth row of inputs
    cols4 = st.columns(3)
    
    # Industrial_Production
    with cols4[0]:
        latest_inputs['Industrial_Production'] = st.number_input(
            label=input_labels['Industrial_Production'],
            min_value=-2.0, 
            max_value=12.0, 
            value=5.37,
            format="%.4f"
        )
        
    # Retail_Sales
    with cols4[1]:
        latest_inputs['Retail_Sales'] = st.number_input(
            label=input_labels['Retail_Sales'],
            min_value=1.0, 
            max_value=14.0, 
            value=8.12,
            format="%.4f"
        )
        
    # Exchange_Rate
    with cols4[2]:
        latest_inputs['Exchange_Rate'] = st.number_input(
            label=input_labels['Exchange_Rate'],
            min_value=60.0, 
            max_value=95.0, 
            value=71.66,
            format="%.4f"
        )

    submitted = st.form_submit_button("Predict CPI for 2024", type="primary")

# --- 4. Prediction Logic ---
if submitted:
    try:
        # Re-map the latest inputs into the correct feature order and format
        # The CPI input becomes CPI_lag1
        latest_data_for_model = {
            'CPI': latest_inputs['CPI'],
            'GDP_Growth': latest_inputs['GDP_Growth'],
            'Unemployment_Rate': latest_inputs['Unemployment_Rate'],
            'Crude_Oil_Prices': latest_inputs['Crude_Oil_Prices'],
            'Gold_Prices': latest_inputs['Gold_Prices'],
            'Real_Estate_Index': latest_inputs['Real_Estate_Index'],
            'Stock_Index': latest_inputs['Stock_Index'],
            'Money_Supply_M1': latest_inputs['Money_Supply_M1'],
            'Money_Supply_M2': latest_inputs['Money_Supply_M2'],
            'Industrial_Production': latest_inputs['Industrial_Production'],
            'Retail_Sales': latest_inputs['Retail_Sales'],
            'Exchange_Rate': latest_inputs['Exchange_Rate']
        }
        
        # We need the values for the 12 features that will be used as input (lagged features)
        # We simulate the process of lagging
        latest_input_row = []
        
        # 1. CPI_lag1 (This is the current year's CPI input)
        latest_input_row.append(latest_data_for_model['CPI'])
        
        # 2-12. The remaining 11 features (lagged from current year)
        for col in ['GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
                    'Gold_Prices', 'Real_Estate_Index', 'Stock_Index', 
                    'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
                    'Retail_Sales', 'Exchange_Rate']:
            latest_input_row.append(latest_data_for_model[col])
            
        # The latest data for the 4th time step (Year 2023)
        latest_time_step = np.array(latest_input_row)

        # The GRU needs 4 time steps (2020, 2021, 2022, 2023 data)
        # X_input sequence shape is (1, 4, 12)
        
        # Combine historical (2020, 2021, 2022) with latest (2023)
        historical_unscaled = HISTORICAL_DF.tail(N_STEPS - 1).values
        full_sequence_unscaled = np.vstack([historical_unscaled, latest_time_step.reshape(1, -1)])

        # Scale the full sequence
        X_scaled = (full_sequence_unscaled - SCALER_X_MEAN) / SCALER_X_STD
        
        # Reshape for GRU: (1, n_steps, n_features)
        X_input = X_scaled.reshape(1, N_STEPS, X_scaled.shape[1])
        
        # Prediction (on scaled output)
        y_pred_s = model.predict(X_input, verbose=0)
        
        # Inverse transform the prediction
        y_pred_raw = y_pred_s[0, 0]
        y_pred = (y_pred_raw * SCALER_Y_STD) + SCALER_Y_MEAN

        # --- Display Results ---
        st.markdown(f"### 📈 Predicted CPI for 2024: <span style='color: #EF4444; font-weight: 800;'>{y_pred:.4f}%</span>", unsafe_allow_html=True)
        
        st.success("Forecast complete! See the detailed inputs used below.")
        
        # Display the 4-year sequence used
        full_sequence_display = pd.DataFrame(full_sequence_unscaled, 
                                             columns=HISTORICAL_DF.columns, 
                                             index=[f"Year {y}" for y in [HISTORICAL_DATA_YEARS[0], HISTORICAL_DATA_YEARS[1], HISTORICAL_DATA_YEARS[2]] + [2023]])
        
        st.markdown("##### Complete 4-Year Input Sequence (Unscaled) for 2024 Forecast")
        st.dataframe(full_sequence_display.iloc[:, :12].style.format("{:.4f}")) # Show all 12 features

        st.info(f"The model's prediction of {y_pred:.4f}% is based on the data above. "
                "The core logic uses the last 4 time steps of these 12 features to forecast the CPI value for the next year (2024).")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the model file is correctly loaded and the input format is consistent.")
        st.exception(e)

# --- Historical Context (Optional but helpful) ---
st.markdown("---")
st.markdown("#### Model Context: Historical Data Used")
st.write("The model was trained on yearly average data from 2008 to 2019. The latest historical values (2020-2022) are used as the start of the prediction sequence.")

st.markdown("##### Last 3 Historical Data Points (2020-2022 Averages)")
st.dataframe(HISTORICAL_DF.style.format("{:.4f}"))

st.markdown("""
<div style="margin-top: 20px; padding: 15px; border: 1px solid #D1D5DB; border-radius: 8px; background-color: #F9FAFB;">
    <p style="font-weight: 600; margin-bottom: 5px;">Model Details:</p>
    <ul>
        <li><b>Architecture:</b> Stacked GRU (16 units, then 8 units) with Dropout and a Dense output layer.</li>
        <li><b>Input Shape:</b> (4 time steps, 12 features)</li>
        <li><b>Prediction:</b> Next Year's CPI (Time step 5)</li>
    </ul>
</div>
""", unsafe_allow_html=True)