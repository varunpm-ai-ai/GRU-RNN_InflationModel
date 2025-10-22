import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

DATA_PATH = '../input/india-inflation-dataset/india_inflation_dataset.csv'
MODEL_NAME = 'india'
MODEL_DIR = './models' 

os.makedirs(MODEL_DIR, exist_ok=True)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from {DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please check your Kaggle dataset path.")
    exit()

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

df_yearly = df.groupby('Year').mean().reset_index()

FEATURE_LAG_COLS = [
    'GDP_Growth', 'Unemployment_Rate', 'Crude_Oil_Prices', 
    'Gold_Prices', 'Real_Estate_Index', 'Stock_Index',
    'Money_Supply_M1', 'Money_Supply_M2', 'Industrial_Production', 
    'Retail_Sales', 'Exchange_Rate'
]
TARGET_COL = 'CPI'
N_STEPS = 4 

def create_sequences_and_scale(df, feature_cols, target_col, n_steps):
    lagged_df = df.copy()
    for col in feature_cols + [target_col]:
        lagged_df[f'{col}_lag1'] = lagged_df[col].shift(1)
        
    lagged_df = lagged_df.dropna().reset_index(drop=True)
    sequence_input_cols = [f'{col}_lag1' for col in feature_cols + [target_col]]
    X_list, y_list = [], []
    data = lagged_df[sequence_input_cols + [target_col]].values

    for i in range(n_steps, len(lagged_df)):
        X_list.append(data[i-n_steps:i, :-1])  
        y_list.append(data[i, -1])            
        
    X_sequences = np.array(X_list)
    y_targets = np.array(y_list)
    
    print(f"Total sequences created. X shape: {X_sequences.shape}, y shape: {y_targets.shape}")

    # Train/Test Split 
    train_size = int(len(X_sequences) * 0.8)
    X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
    y_train, y_test = y_targets[:train_size], y_targets[train_size:]
    
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    scaler_X = StandardScaler()
    X_train_s_2d = scaler_X.fit_transform(X_train_2d)
    X_test_s_2d = scaler_X.transform(X_test_2d)
    
    X_train_s = X_train_s_2d.reshape(X_train.shape)
    X_test_s = X_test_s_2d.reshape(X_test.shape)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print("Data successfully scaled.")
    
    return X_train_s, X_test_s, y_train_s, y_test_s, y_test, scaler_X, scaler_y, train_size

X_train_s, X_test_s, y_train_s, y_test_s, y_test_actual, scaler_X, scaler_y, train_size = create_sequences_and_scale(
    df_yearly, FEATURE_LAG_COLS, TARGET_COL, N_STEPS
)

# GRU RNN Model
Sequential = tf.keras.models.Sequential
GRU = tf.keras.layers.GRU
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
l2 = tf.keras.regularizers.l2

model = Sequential([
    GRU(16, activation='tanh', return_sequences=True, input_shape=(X_train_s.shape[1], X_train_s.shape[2])),
    Dropout(0.2),
    GRU(8, activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
print("\n--- Model Architecture ---")
model.summary()

# MODEL TRAINING 
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

print("\n--- Starting Training ---")
history = model.fit(
    X_train_s, y_train_s,
    epochs=500,
    batch_size=4,
    validation_data=(X_test_s, y_test_s),
    callbacks=[early_stop, reduce_lr],
    verbose=2
)
print("Training complete.")


# Make predictions (scaled)
y_pred_s = model.predict(X_test_s, verbose=0)
y_pred_actual = scaler_y.inverse_transform(y_pred_s).flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n--- Model Performance on Test Set ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save the trained model
model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_gru_model.h5')
model.save(model_path)
print(f"\nSaved GRU Model: {model_path}")

scaler_X_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_scaler_X.joblib')
joblib.dump(scaler_X, scaler_X_path)
print(f"Saved Input Scaler (X): {scaler_X_path}")

scaler_y_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_scaler_y.joblib')
joblib.dump(scaler_y, scaler_y_path)
print(f"Saved Target Scaler (y): {scaler_y_path}")

years_test = df_yearly['Year'].iloc[N_STEPS + train_size + 1:].values
cmp_df = pd.DataFrame({
    'Year': years_test,
    'Actual_CPI': y_test_actual,
    'Predicted_CPI': y_pred_actual
}).reset_index(drop=True)
print("\n--- Actual vs. Predicted (Test Set) ---")
print(cmp_df)