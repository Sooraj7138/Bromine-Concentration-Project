import os
import numpy as np
import pandas as pd
import joblib
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# === Constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Bromine_Prediction_Dataset_new.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'waterlevel_prediction_output.csv')

FEATURES = ['Day', 'Month', 'Year', 'Month_sin', 'Month_cos']
TARGET = 'Wl'
MODEL_NAME = 'xgb_waterlevel_model'

# === Helper Function ===
def preprocess_dates(df):
    # ✅ Ensure 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # coerce invalid dates to NaT
    if df['Date'].isnull().any():
        raise ValueError("Some values in 'Date' column could not be parsed to datetime. Please check your data.")

    # ✅ Then proceed with datetime-based features
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    return df

def robust_scale_y(y_raw):
    y_clean = y_raw[(y_raw > np.percentile(y_raw, 1)) & (y_raw < np.percentile(y_raw, 99))]
    scaler = MinMaxScaler()
    scaler.fit(y_clean.reshape(-1, 1))
    y_scaled = scaler.transform(y_raw.reshape(-1, 1))
    return scaler, y_scaled

# === Model Training Function ===
def train_model(df_clean, y_column, model_name, scaler_x):
    print(f"\n🔧 Training XGBoost model for: {y_column}")
    y_raw = df_clean[y_column].values.reshape(-1, 1)
    if np.isnan(y_raw).any():
        raise ValueError(f"❌ Target column '{y_column}' contains NaNs.")

    # Scale target (Y)
    scaler_y, y_scaled = robust_scale_y(y_raw)

    # Scale input (X)
    X = df_clean[FEATURES].values
    X_scaled = scaler_x.transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

    # Train XGBoost model
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train.ravel())

    # Save model and scaler
    model_path = os.path.abspath(os.path.join(MODEL_DIR, f"{model_name}_wl.h5"))
    joblib.dump(model, model_path)
    joblib.dump(scaler_y, os.path.abspath(os.path.join(MODEL_DIR, f'{model_name}_scaler_wl.pkl')))
    print(f"✅ Saved XGBoost model and scaler: {model_name}")

# === Train Model ===
def train_waterlevel_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = preprocess_dates(df)

    required_columns = [TARGET] + FEATURES
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns in dataset: {missing_cols}")

    df_clean = df.dropna(subset=required_columns)
    if df_clean.empty:
        raise ValueError("❌ No valid rows found in dataset after cleaning.")

    X = df_clean[FEATURES].values
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    joblib.dump(scaler_x, os.path.join(MODEL_DIR, 'x_scaler_wl.pkl'))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'x_features_wl.pkl'))
    print("✅ Saved input scaler and feature list.")

    train_model(df_clean, TARGET, MODEL_NAME, scaler_x)

    print("\n🎉 Water level model trained and saved.")
    feature_ranges = {}
    for i, feature in enumerate(FEATURES):
        feature_ranges[feature] = {
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max())
        }
    joblib.dump(feature_ranges, os.path.join(MODEL_DIR, 'x_feature_ranges_wl.pkl'))
    print("✅ Saved input feature value ranges for validation.")

# === Forecast Function ===
def forecast_future_waterlevel(start_date_str, end_date_str):
    df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)

    if 'Date' not in df_hist.columns:
        raise ValueError("❌ 'Date' column missing from dataset.")

    if df_hist['Date'].isnull().all():
        raise ValueError("❌ 'Date' column has only null or invalid values.")

    last_date = df_hist['Date'].dropna().max()
    if pd.isna(last_date):
        raise ValueError("❌ Could not determine last date from the dataset.")

    start_date = pd.to_datetime(start_date_str + "-01", format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date_str + "-01", format="%Y-%m-%d") + pd.offsets.MonthEnd(0)

    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_future = pd.DataFrame({'Date': future_dates})
    df_future = preprocess_dates(df_future)

    model = joblib.load(os.path.join(MODEL_DIR, f'{MODEL_NAME}_wl.h5'))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, f'{MODEL_NAME}_scaler_wl.pkl'))
    scaler_x = joblib.load(os.path.join(MODEL_DIR, 'x_scaler_wl.pkl'))
    feature_ranges = joblib.load(os.path.join(MODEL_DIR, 'x_feature_ranges_wl.pkl'))

    X = df_future[FEATURES].values
    for i, feature in enumerate(FEATURES):
        f_min = feature_ranges[feature]['min']
        f_max = feature_ranges[feature]['max']
        if X[:, i].min() < f_min or X[:, i].max() > f_max:
            print(f"⚠️ Feature {feature} out of training range: input {X[:, i].min()} – {X[:, i].max()} | trained range: {f_min} – {f_max}")

    X_scaled = scaler_x.transform(X)
    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    df_future[TARGET] = pred

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_future.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Water level forecast complete and saved to:\n{OUTPUT_PATH}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future[TARGET],
                             mode='lines', name='Predicted Water Level'))

    fig.update_layout(
        title='Predicted Water Level (m)',
        xaxis_title='Date',
        yaxis_title='Water Level (m)',
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white',
        height=600,
        width=1800
    )

    plot_html = fig.to_html(full_html=False)
    df_future['Water Level (m)'] = df_future[TARGET]
    return df_future[['Date', 'Water Level (m)']], plot_html

# === Entry Point ===
if __name__ == "__main__":
    train_waterlevel_model()
    # forecast_future_waterlevel(start_date_str = '2024-01', end_date_str = '2024-12')
