import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Inflation GRU Demo", layout="wide")

st.title("Inflation GRU — Mini Project Demo")
st.markdown("Select model (India / Bangalore), provide a dataset (CSV) or use a default, then compare Actual vs Predicted CPI.")

# --- model & scaler mapping (adjust filenames if yours differ) ---
MODEL_DIR = Path("models")
model_map = {
    "India (gru_model.h5)": {
        "model": MODEL_DIR / "gru_model.h5",
        "scaler": MODEL_DIR / "scaler.save"
    },
    "Bangalore (gru_model2.h5)": {
        "model": MODEL_DIR / "gru_model2.h5",
        "scaler": MODEL_DIR / "scaler3.save"
    }
}

# Sidebar controls
with st.sidebar:
    choice = st.selectbox("Choose model", list(model_map.keys()))
    use_uploaded = st.checkbox("Upload CSV (override default)", value=False)
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) if use_uploaded else None
    default_path = st.text_input("Or dataset path (relative or absolute)",
                                 value="datasets/Bengaloredataset/bangalore_inflation_dataset.csv")
    n_steps = st.number_input("Sequence length (n_steps)", min_value=1, max_value=10, value=3, step=1)
    show_metrics = st.checkbox("Show metrics", value=True)
    st.markdown("---")
    st.markdown("Notes:")
    st.caption("• Models expected in `models/` folder. Filenames in mapping can be edited in the code.")
    st.caption("• CSV must include a 'Year' column and numeric columns: 'CPI' and factor columns (Gold_Prices, Crude_Oil_Prices, Real_Estate_Index, Stock_Index).")

# load dataset
def load_dataframe():
    if uploaded is not None:
        return pd.read_csv(uploaded)
    p = Path(default_path)
    if not p.exists():
        # try from project root
        p2 = Path.cwd() / default_path
        if p2.exists():
            p = p2
    if p.exists():
        return pd.read_csv(p)
    st.error(f"Dataset not found: {p}")
    return None

df = load_dataframe()
if df is None:
    st.stop()

# basic cleaning & checks
if "Year" not in df.columns:
    st.error("CSV must contain a 'Year' column.")
    st.stop()
if "CPI" not in df.columns:
    st.error("CSV must contain a 'CPI' column.")
    st.stop()

df = df.sort_values("Year").reset_index(drop=True)
years = df["Year"].values

# try load model & scaler
model_path = model_map[choice]["model"]
scaler_path = model_map[choice]["scaler"]

@st.cache_resource
def load_keras_model(p: str):
    try:
        return tf.keras.models.load_model(p)
    except Exception as e:
        return None

@st.cache_resource
def load_scaler(p: str):
    try:
        return joblib.load(p)
    except Exception:
        try:
            # try pickle
            import pickle
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

model = load_keras_model(str(model_path))
scaler = load_scaler(str(scaler_path))

if model is None:
    st.error(f"Could not load model from {model_path}. Check filename/path.")
    st.stop()

# prepare sequences using numeric columns except Year; target = CPI
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# ensure CPI included; remove Year
if "Year" in numeric_cols:
    numeric_cols.remove("Year")
features = [c for c in numeric_cols if c != "CPI"]
if "CPI" not in numeric_cols:
    st.error("No numeric 'CPI' column found after type inference.")
    st.stop()

# build sliding windows for features; y is CPI at time t
vals = df[features].values
cpi_vals = df["CPI"].values
X, y, idx_years = [], [], []
for i in range(n_steps, len(df)):
    X.append(vals[i - n_steps:i])       # shape (n_steps, n_features)
    y.append(cpi_vals[i])
    idx_years.append(df["Year"].iloc[i])
if len(X) == 0:
    st.error("Not enough rows for the chosen n_steps. Reduce n_steps.")
    st.stop()
X = np.array(X)
y = np.array(y)

# model predict (model may expect different feature layout — we attempt to call)
try:
    y_pred_s = model.predict(X, verbose=0)
    # ensure shape
    y_pred_s = np.array(y_pred_s).reshape(-1)
except Exception as e:
    st.error(f"Model prediction failed: {e}")
    st.stop()

# inverse transform if scaler available and has inverse_transform
if scaler is not None:
    try:
        y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(-1)
    except Exception:
        # scaler not for target or incompatible; fallback to raw
        y_pred = y_pred_s
else:
    y_pred = y_pred_s

# Layout: two columns for plot + metrics
col1, col2 = st.columns([3,1])

with col1:
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(idx_years, y, marker="o", lw=2, color="tab:blue", label="Actual CPI")
    ax.plot(idx_years, y_pred, marker="x", lw=2, color="tab:orange", label="Predicted CPI")
    ax.set_xlabel("Year")
    ax.set_ylabel("CPI")
    ax.set_title(f"Actual vs Predicted CPI — {choice.split()[0]}")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # option: compare a factor selected by user (actual only)
    st.markdown("### Compare factor with CPI")
    factor = st.selectbox("Show factor", ["Gold_Prices", "Crude_Oil_Prices", "Real_Estate_Index", "Stock_Index"])
    if factor in df.columns:
        fig2, ax1 = plt.subplots(figsize=(10,3.5))
        ax1.plot(df["Year"].iloc[n_steps:], y, marker="o", color="tab:blue", label="Actual CPI")
        ax1.plot(df["Year"].iloc[n_steps:], y_pred, marker="x", color="tab:orange", label="Predicted CPI")
        ax2 = ax1.twinx()
        ax2.plot(df["Year"].iloc[n_steps:], df[factor].iloc[n_steps:], marker="s", color="tab:green", label=factor)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("CPI")
        ax2.set_ylabel(factor)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.info(f"Factor column '{factor}' not found in dataset. Available numeric columns: {numeric_cols}")

with col2:
    st.subheader("Quick metrics")
    if show_metrics:
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R2", f"{r2:.4f}")
        st.write("MSE:", f"{mse:.4f}")
    st.markdown("### Model files")
    st.write(f"Model: {model_path}")
    st.write(f"Scaler: {scaler_path} (loaded: {scaler is not None})")

st.markdown("---")
st.caption("App created for a college mini project. Edit model paths in app.py if your filenames differ.")
