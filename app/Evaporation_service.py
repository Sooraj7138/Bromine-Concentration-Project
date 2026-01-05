import os
import numpy as np
import pandas as pd
import joblib
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# === Constants ===
GAMMA = 67.4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Main Dataset for Wind 2022.csv')

OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'evaporation_prediction_output.csv')
print(OUTPUT_PATH)

FEATURES = ['Day', 'Month', 'Year', 'Month_sin', 'Month_cos']

TARGETS = {
    'Solar-Irradiance': 'xgb_ra_model',
    'Slope': 'xgb_slope_model',
    'Saturation_vapour': 'xgb_es_model',
    'Actual_vapour': 'xgb_ea_model',
    'wind_speed': 'xgb_wind_model',
    'Radiation-Based_evaporation': 'xgb_er_model',
    'Aerodynamics_evaporation': 'xgb_ea_calc_model',
    'Psychrometric constant': 'xgb_gamma_model'  # Now predicted!
}

# === Helper Function ===
def preprocess_dates(df):
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
    model_path = os.path.abspath(os.path.join(MODEL_DIR, f"{model_name}.h5"))
    joblib.dump(model, model_path)
    joblib.dump(scaler_y, os.path.abspath(os.path.join(MODEL_DIR, f'{model_name}_scaler.pkl')))
    print(f"✅ Saved XGBoost model and scaler: {model_name}")

# === Train All Models ===
def train_all_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = preprocess_dates(df)

    required_columns = list(TARGETS.keys()) + FEATURES
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns in dataset: {missing_cols}")

    df_clean = df.dropna(subset=required_columns)

    if df_clean.empty:
        raise ValueError("❌ No valid rows found in dataset after cleaning.")

    # Save and apply X scaler
    X = df_clean[FEATURES].values
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    joblib.dump(scaler_x, os.path.join(MODEL_DIR, 'x_scaler.pkl'))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'x_features.pkl'))
    print("✅ Saved input scaler and feature list.")

    for target, model_name in TARGETS.items():
        train_model(df_clean, target, model_name, scaler_x)

    print("\n🎉 All models trained and saved.")
    # Save value ranges for validation
    feature_ranges = {}
    for i, feature in enumerate(FEATURES):
        feature_ranges[feature] = {
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max())
        }
    joblib.dump(feature_ranges, os.path.join(MODEL_DIR, 'x_feature_ranges.pkl'))
    print("✅ Saved input feature value ranges for validation.")

def learn_and_save_ea_scaling_constant():
    df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df_hist = df_hist.dropna(subset=['wind_speed', 'Saturation_vapour', 'Actual_vapour', 'Ea'])

    z2, z0 = 200, 0.04
    log_term_sq = np.log(z2 / z0) ** 2
    df_hist['Ea_base'] = ((0.102 * df_hist['wind_speed']) / log_term_sq) * (
            df_hist['Saturation_vapour'] - df_hist['Actual_vapour']
    )

    X = df_hist[['Ea_base']].values
    y = df_hist['Ea'].values
    model = LinearRegression().fit(X, y)
    K = model.coef_[0]
    joblib.dump(K, os.path.join(MODEL_DIR, 'ea_scaling_factor.pkl'))
    print(f"✅ Learned and saved Ea scaling factor K = {K:.4f}")

# === Forecast Function ===
def forecast_future_days(start_date_str, end_date_str, mode='a', evaptype=None):
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

    # Load models and scalers
    models = {
        target: joblib.load(os.path.join(MODEL_DIR, f'{model_name}.h5'))
        for target, model_name in TARGETS.items()
    }
    scalers = {
        target: joblib.load(os.path.join(MODEL_DIR, f'{model_name}_scaler.pkl'))
        for target, model_name in TARGETS.items()
    }
    scaler_x = joblib.load(os.path.join(MODEL_DIR, 'x_scaler.pkl'))
    feature_ranges = joblib.load(os.path.join(MODEL_DIR, 'x_feature_ranges.pkl'))

    # Prepare input features
    X = df_future[FEATURES].values

    # Optional: Warn if input features are out of training range
    for i, feature in enumerate(FEATURES):
        f_min = feature_ranges[feature]['min']
        f_max = feature_ranges[feature]['max']
        if X[:, i].min() < f_min or X[:, i].max() > f_max:
            print(f"⚠️ Feature {feature} out of training range: input {X[:, i].min()} – {X[:, i].max()} | trained range: {f_min} – {f_max}")

    X_scaled = scaler_x.transform(X)

    # Make predictions
    for target, model in models.items():
        pred_scaled = model.predict(X_scaled)
        pred = scalers[target].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        df_future[target] = pred

    # Compute Evaporation using predicted γ, Er, Ea
    delta = df_future['Slope']
    gamma = df_future['Psychrometric constant']
    Er = df_future['Radiation-Based_evaporation']
    Ea = df_future['Aerodynamics_evaporation']

    df_future['Evaporation'] = (delta / (delta + gamma)) * Er + \
                                (gamma / (delta + gamma)) * Ea
    # ✅ Apply cumulative sum
    df_future['Evaporation'] = df_future['Evaporation'] if mode == 'a' else df_future['Evaporation'].cumsum()
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_future.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Forecast for next days complete and saved to:\n{OUTPUT_PATH}")

    # === Plotting historical and predicted Evaporation & Wind Speed ===
    # df_hist_plot = df_hist[['Date', 'Evaporation', 'wind_speed']].dropna()
    # df_hist_plot = df_hist_plot[df_hist_plot['Date'] <= last_date]

    fig = go.Figure()

    # Evaporation
    # fig.add_trace(go.Scatter(x=df_hist_plot['Date'], y=df_hist_plot['Evaporation'],
    #                          mode='lines', name='Historical Evaporation'))
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Evaporation'],
                             mode='lines', name='Predicted Evaporation'))

    # Wind Speed
    # fig.add_trace(go.Scatter(x=df_hist_plot['Date'], y=df_hist_plot['wind_speed'],
    #                          mode='lines', name='Historical Wind Speed'))
    # fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['wind_speed'],
    #                          mode='lines', name='Predicted Wind Speed'))

    fig.update_layout(
        title=f'Predicted Evaporation Rate (mm/day)',
        xaxis_title='Date',
        yaxis_title='Evaporation Rate (mm/day)',
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white',
        height=600,
        width=1800
    )

    plot_html = fig.to_html(full_html=False)

    df_future['Evaporation (mm/day)'] = df_future['Evaporation']
    df_future['Slope (kPa/°C)'] = df_future['Slope']
    df_future['Psychrometric constant (kPa/°C)'] = df_future['Psychrometric constant']
    df_future['Radiation Based Evaporation (mm/day)'] = df_future['Radiation-Based_evaporation']
    df_future['Aerodynamics Evaporation (mm/day)'] = df_future['Aerodynamics_evaporation']
    df_future['Saturation Vapour (kPa)'] = df_future['Saturation_vapour']
    df_future['Actual Vapour (kPa)'] = df_future['Actual_vapour']
    df_future['Solar Irradiance (MJ m2/day)'] = df_future['Solar-Irradiance']
    df_future['Wind Speed (m/s)'] = df_future['wind_speed']

    return df_future[['Date', 'Evaporation (mm/day)', 'Slope (kPa/°C)', 'Psychrometric constant (kPa/°C)',
                          'Radiation Based Evaporation (mm/day)', 'Aerodynamics Evaporation (mm/day)',
                          'Saturation Vapour (kPa)', 'Actual Vapour (kPa)', 'Solar Irradiance (MJ m2/day)', 'Wind Speed (m/s)']], plot_html

# === Entry Point ===
if __name__ == "__main__":
    train_all_models()
