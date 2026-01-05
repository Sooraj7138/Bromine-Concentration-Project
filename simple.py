import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.regularizers import l2
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import joblib, random
from datetime import timedelta

# Constants
A = 14.062
B = 44.722
C = 56.651
D = 34.955

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds(42)

# Helper: Fix date format from "01-06-2023 0.00" → "01-06-2023 00:00"
def fix_date_format(date_str):
    parts = re.split(r"[.\s]", str(date_str))
    if len(parts) >= 2:
        try:
            hour = int(float(parts[1]))
            return f"{parts[0]} {hour:02d}:00"
        except:
            return date_str
    return date_str

# Detect frequency: hourly, daily, monthly
def detect_frequency(df):
    diffs = df['Date'].sort_values().diff().dropna()
    avg_diff = diffs.mean()
    if avg_diff <= pd.Timedelta(hours=1):
        return 'hourly'
    elif avg_diff <= pd.Timedelta(days=1):
        return 'daily'
    else:
        return 'monthly'

# Normalize
def normalize_data(df_evap, df_wl):
    freq = detect_frequency(df_evap)

    if freq == 'hourly':
        df_evap = df_evap.set_index('Date').resample('D').sum().reset_index()
        df_wl = df_wl.set_index('Date').resample('D').mean().reset_index()
    elif freq == 'monthly':
        df_evap = df_evap.set_index('Date').resample('M').mean().reset_index()
        df_wl = df_wl.set_index('Date').resample('M').mean().reset_index()

    return df_evap, df_wl, freq

def normalize_data(df_evap, df_wl=None):
    freq = detect_frequency(df_evap)

    df_evap = df_evap.set_index('Date')

    if freq == 'hourly':
        df_evap = df_evap.resample('D').sum().reset_index()
        if df_wl is not None:
            df_wl = df_wl.set_index('Date').resample('D').mean().reset_index()
    elif freq == 'monthly':
        df_evap = df_evap.resample('M').mean().reset_index()
        if df_wl is not None:
            df_wl = df_wl.set_index('Date').resample('M').mean().reset_index()
    else:
        df_evap = df_evap.reset_index()
        if df_wl is not None:
            df_wl = df_wl.reset_index()

    if df_wl is not None:
        return df_evap, df_wl, freq
    return df_evap, freq

# Volume and concentration logic
def calculate_volume(wl):
    return A * wl**3 + B * wl**2 + C * wl + D

def calculate_surface_area(wl):
    return 3 * A * wl + 2 * B * wl + C

def calculate_concentration(wl_df, evap_df, in_con, mode='a', unit=None, pond_name=None):
    if mode == 'a':
        # 🔹 ZONE MODE: Requires wl_df
        df = pd.merge(wl_df, evap_df, on='Date', how='inner')
        df = df.dropna(subset=['Wl', 'Evaporation'])

        df['Volume'] = calculate_volume(df['Wl'])
        df['SurfaceArea'] = calculate_surface_area(df['Wl'])
        if unit == 'wt%':
            df['EvapVol'] = df['Evaporation'] * df['SurfaceArea'] / 1000  # mm * m² → m³
            df['V2'] = df['Volume'] - df['EvapVol']
            print(in_con)
            df['C2'] = (df['Volume'] * in_con) / df['V2']
            return df[['Date', 'Wl', 'Evaporation', 'Volume', 'SurfaceArea', 'EvapVol', 'C2']]
        else:
            df['Water_Evaporated'] = df['Evaporation'] * df['SurfaceArea']
            df['C2'] = (in_con / (100 - in_con)) * df['Water_Evaporated']
            print(in_con)
            return df[['Date', 'Wl', 'Evaporation', 'Volume', 'SurfaceArea', 'Water_Evaporated', 'C2']]

    else:
        pond_volumes = {
            'A': 584994.6,
            'B': 558427.3,
            'C': 478725.2,
            'D': 399030.1,
            'E': 319476.2,
            'F': 240507.2
        }
        # 🔹 POND MODE: No wl_df needed
        df = evap_df.copy()
        df = df.dropna(subset=['Evaporation'])
        volume_m3 = pond_volumes.get(pond_name.upper(), 584994.6)  # Default to Pond A if not found
        df['Volume_m3'] = volume_m3 # fixed volume in m³
        print("pond name : ",pond_name)
        df['SurfaceArea'] = 9       # fixed surface area in m²
        df['Volume'] = df['Volume_m3'] * 1000  # Convert from ML to m³
        if unit == 'wt%':
            df['EvapVol'] = df['Evaporation'] * df['SurfaceArea'] / 1000
            df['V2'] = df['Volume'] - df['EvapVol']
            print(in_con)
            df['C2 (wt%)'] = (df['Volume'] * in_con) / df['V2']
            return df[['Date', 'Evaporation', 'Volume', 'SurfaceArea', 'EvapVol', 'C2 (wt%)']]


# Plotting
def plot_concentration_with_forecast(category, forecast_df, unit=None):
    # === Load historical C2 from saved file ===
    df_path = "outputs/bromine_concentration_output.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError("⚠ Historical C2 data file not found.")

    historical_df = pd.read_csv(df_path, parse_dates=["Date"])
    historical_df = historical_df[['Date', 'C2 (wt%)']].dropna()
    historical_df = historical_df[(historical_df['Date'] >= '2023-01-01') & (historical_df['Date'] <= '2023-12-31')]

    # === Plotting ===
    fig = go.Figure()

    # Historical Trace
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['C2 (wt%)'],
        mode='lines',
        name='Historical C2 (Jun–Dec 2023)',
        line=dict(color='blue')
    ))

    # Forecast Trace
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['C2 (wt%)'],
        mode='lines',
        name='Forecasted C2 (2024)',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Bromine Concentration: Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title="C2 (wt%)",
        template="plotly_white",
        height=600,
        width=1800
    )

    return fig.to_html(full_html=False)

def predict_future_concentration_gru(user_start, user_end, mode='a', unit=None, pond_name=None):
    forecast_days = 365
    lag_days = 120 if unit == 'wt%' else 30
    sequence_length = 7  # Not used in XGBoost, kept for compatibility

    # === File paths ===
    if mode == 'b' and pond_name:
        pond_key = pond_name.upper()
        df_path = f"outputs/bromine_concentration_output_pond_{pond_key}_{unit}.csv"
        model_path = f"models/xgb_c2_model_pond_{pond_key}.json"
        scaler_path = f"models/xgb_c2_scaler_pond_{pond_key}.save"
    else:
        df_path = "outputs/bromine_concentration_output.csv"
        model_path = "models/xgb_c2_model.json"
        scaler_path = "models/xgb_c2_scaler.save"

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"❌ Required C2 data file not found at: {df_path}")

    # === Load and preprocess ===
    df = pd.read_csv(df_path, parse_dates=["Date"])
    if mode == 'a':
        if unit == 'wt%':
            df = df[["Date", "C2", "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]].dropna()
        else:
            df = df[["Date", "C2", "Wl", "Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"]].dropna()
    else:
        if unit == 'wt%':
            df = df[["Date", "C2 (wt%)", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]].dropna()
            df['C2'] = df['C2 (wt%)']
        else:
            df = df[["Date", "C2 (kg)", "Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"]].dropna()
            df['C2'] = df['C2 (kg)']
    df = df.set_index("Date").asfreq("D").interpolate(method='time')

    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["trend"] = np.arange(len(df))
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # === Rolling features ===
    rolling_windows = [14, 30, 60] if unit == 'wt%' else [14]
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["C2"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["C2"].rolling(window=window).std()
    df["cumulative_mean"] = df["C2"].expanding().mean()
    df["cumulative_std"] = df["C2"].expanding().std()

    # === Lag features ===
    for i in range(1, lag_days + 1):
        df[f"lag_{i}"] = df["C2"].shift(i)

    df.dropna(inplace=True)

    # === Features ===
    if unit == 'wt%':
        feature_cols = [
                           "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                           "cumulative_mean", "cumulative_std",
                           "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"
                       ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                             rolling_windows] \
                       + [f"lag_{i}" for i in range(1, lag_days + 1)] if mode == 'a' else [
                           "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                           "cumulative_mean", "cumulative_std",
                           "Evaporation", "Volume", "SurfaceArea", "EvapVol"
                       ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                             rolling_windows] \
                       + [f"lag_{i}" for i in range(1, lag_days + 1)]
    else:
        feature_cols = [
                           "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                           "cumulative_mean", "cumulative_std",
                           "Wl", "Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"
                       ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                             rolling_windows] \
                       + [f"lag_{i}" for i in range(1, lag_days + 1)] if mode == 'a' else [
                           "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                           "cumulative_mean", "cumulative_std",
                           "Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"
                       ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                             rolling_windows] \
                       + [f"lag_{i}" for i in range(1, lag_days + 1)]

    X = df[feature_cols].values
    y = df["C2"].values

    # === Scale ===
    os.makedirs("models", exist_ok=True)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        try:
            scaler.transform(X)
        except ValueError:
            print("⚠️ Feature mismatch. Re-fitting scaler.")
            scaler = MinMaxScaler()
            scaler.fit(X)
            joblib.dump(scaler, scaler_path)
            if os.path.exists(model_path):
                os.remove(model_path)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        joblib.dump(scaler, scaler_path)

    X_scaled = scaler.transform(X)

    # === Train XGBoost ===
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9) if unit == 'wt%' else XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1
    )
    model.fit(X_scaled, y)

    # === Forecasting ===
    last_known_date = df.index[-1]
    past_c2 = df["C2"].iloc[-lag_days:].tolist()
    if unit == 'wt%':
        series_dict = {col: df[col] for col in ["Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]} if mode == 'a' else {col: df[col] for col in ["Evaporation", "Volume", "SurfaceArea", "EvapVol"]}
    else:
        series_dict = {col: df[col] for col in ["Wl", "Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"]} if mode == 'a' else {col: df[col] for col in ["Evaporation", "Volume", "SurfaceArea", "Water_Evaporated"]}
    future_dates = []
    future_c2 = []

    for day in range(1, forecast_days + 1):
        next_date = last_known_date + timedelta(days=day)
        dayofyear = next_date.dayofyear
        month = next_date.month
        trend = len(df) + day - 1
        dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365)
        dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        rolling_means = [np.mean(past_c2[-w:]) for w in rolling_windows]
        rolling_stds = [np.std(past_c2[-w:]) for w in rolling_windows]
        cumulative_mean = np.mean(past_c2)
        cumulative_std = np.std(past_c2)

        # Reuse historical series cyclically to prevent flattening
        def get_series_val(s, idx):
            try:
                return s.loc[next_date]
            except KeyError:
                return s.iloc[idx % len(s)]

        if mode == 'a':
            wl = get_series_val(series_dict["Wl"], day)
        else: None
        evaporation = get_series_val(series_dict["Evaporation"], day)
        volume = get_series_val(series_dict["Volume"], day)
        surface_area = get_series_val(series_dict["SurfaceArea"], day)
        if unit == 'wt%':
            evapvol = get_series_val(series_dict["EvapVol"], day)
            lag_features = past_c2[-lag_days:]

            raw_input_row = np.array([[dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                                       cumulative_mean, cumulative_std,
                                       wl, evaporation, volume, surface_area, evapvol] +
                                      rolling_means + rolling_stds + lag_features]) if mode == 'a' else np.array([[dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                                       cumulative_mean, cumulative_std,
                                       evaporation, volume, surface_area, evapvol] +
                                      rolling_means + rolling_stds + lag_features])
        else:
            Water_Evaporated = get_series_val(series_dict["Water_Evaporated"], day)
            lag_features = past_c2[-lag_days:]

            raw_input_row = np.array([[dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                                       cumulative_mean, cumulative_std,
                                       wl, evaporation, volume, surface_area, Water_Evaporated] +
                                      rolling_means + rolling_stds + lag_features]) if mode == 'a' else np.array([[dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                                       cumulative_mean, cumulative_std,
                                       evaporation, volume, surface_area, Water_Evaporated] +
                                      rolling_means + rolling_stds + lag_features])

        if raw_input_row.shape[1] != scaler.n_features_in_:
            raise ValueError(f"❌ Feature count mismatch: {raw_input_row.shape[1]} vs expected {scaler.n_features_in_}")

        input_scaled = scaler.transform(raw_input_row)
        pred = model.predict(input_scaled)[0]

        future_c2.append(pred)
        future_dates.append(next_date)
        past_c2.append(pred)

    forecast_df = pd.DataFrame({"Date": future_dates, "C2 (wt%)": future_c2}) if unit == 'wt%' else pd.DataFrame({"Date": future_dates, "C2 (kg)": future_c2})
    forecast_df = forecast_df.set_index("Date").loc[user_start:user_end].reset_index()

    return forecast_df, df.reset_index()

# === Main Pipeline ===
def plot_concentration(result_df, unit=None):
    # === Plotting ===
    fig = go.Figure()

    # Forecast Trace
    fig.add_trace(go.Scatter(
        x=result_df['Date'],
        y=result_df['C2 (wt%)'] if unit == 'wt%' else result_df['C2 (kg)'],
        mode='lines',
        name='Forecasted C2 (2024)',
        line=dict(color='blue')
    ))

    if unit == 'wt%':
        fig.update_layout(
            title="Bromine Concentration: Historical vs Forecast",
            xaxis_title="Date",
            yaxis_title="C2 (wt%)",
            template="plotly_white",
            height=600,
            width=1800
        )
    else:
        fig.update_layout(
            title="Bromine Concentration: Historical vs Forecast",
            xaxis_title="Date",
            yaxis_title="C2 (kg)",
            template="plotly_white",
            height=600,
            width=1800
        )

    return fig.to_html(full_html=False)


def run_bromine_concentration_pipeline(wl_path, evap_path, category, in_con, mode, unit=None, pond_name=None, forecast=True, user_start=None, user_end=None):
    print(f"⚙ Mode selected: {mode}")
    try:
        # Pre-check frequency using a light read
        temp_evap = pd.read_csv(evap_path, usecols=['Date'])
        temp_evap['Date'] = temp_evap['Date'].apply(fix_date_format)
        temp_evap['Date'] = pd.to_datetime(temp_evap['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
        is_hourly = detect_frequency(temp_evap) == 'hourly'
    except Exception as e:
        print("⚠ Frequency detection failed:", e)
        is_hourly = False

    try:
        if mode == 'a':
            # === ZONE MODE ===
            if is_hourly:
                wl_df = pd.read_csv(wl_path)
                evap_df = pd.read_csv(evap_path)
                wl_df['Date'] = wl_df['Date'].apply(fix_date_format)
                evap_df['Date'] = evap_df['Date'].apply(fix_date_format)
                wl_df['Date'] = pd.to_datetime(wl_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
                evap_df['Date'] = pd.to_datetime(evap_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
            else:
                wl_df = pd.read_csv(wl_path, parse_dates=['Date'], dayfirst=True)
                evap_df = pd.read_csv(evap_path, parse_dates=['Date'], dayfirst=True)

            evap_df, wl_df, freq = normalize_data(evap_df, wl_df)
            result_df = calculate_concentration(wl_df, evap_df, in_con, unit=unit, mode=mode)

            # 🟦 Save calculated C2
            output_path = os.path.join("outputs", "bromine_concentration_output.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 GRU Forecast only for ZONE
            if forecast and user_start and user_end:
                forecast_df, _ = predict_future_concentration_gru(user_start, user_end, unit=unit)
                forecast_df.to_csv("outputs/forecasted_c2.csv", index=False)
                plot_html = plot_concentration_with_forecast(category, forecast_df ,unit=unit)
                return forecast_df, plot_html

            # If no forecast requested
            plot_html = plot_concentration_with_forecast(category , result_df ,unit=unit)
            return result_df, plot_html

        else:
            # === POND MODE === (No GRU prediction)
            if is_hourly:
                evap_df = pd.read_csv(evap_path)
                evap_df['Date'] = evap_df['Date'].apply(fix_date_format)
                evap_df['Date'] = pd.to_datetime(evap_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
            else:
                evap_df = pd.read_csv(evap_path, parse_dates=['Date'], dayfirst=True)

            evap_df, freq = normalize_data(evap_df)
            result_df = calculate_concentration(None, evap_df, in_con, mode=mode, unit=unit, pond_name=pond_name)

            # Save result for continuity if needed
            output_path = os.path.join("outputs", f"bromine_concentration_output_pond_{pond_name}_{unit}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 Forecast (same XGBoost model)
            if forecast and user_start and user_end:
                forecast_df, _ = predict_future_concentration_gru(user_start, user_end, mode='b',pond_name=pond_name, unit=unit)
                forecast_df.to_csv(f"outputs/forecasted_c2_pond_{pond_name}_{unit}.csv", index=False)
                plot_html = plot_concentration(forecast_df, unit=unit)
                return forecast_df, plot_html

            plot_html = plot_concentration(result_df, unit=unit)
            return result_df, plot_html

    except Exception as e:
        print("❌ Prediction failed:", e)
        raise

if __name__ == "__main__":
    # predict_future_concentration_gru(user_start='01-2024',user_end='12-2024')
    predict_future_concentration_gru(user_start='01-2024', user_end='12-2024', mode='b', pond_name='B')



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
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Main Dataset for Wind.csv')

OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'evaporation_prediction_output.csv')

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
def forecast_future_days(num_days):
    df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    last_date = df_hist['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days)
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

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_future.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Forecast for next {num_days} days complete and saved to:\n{OUTPUT_PATH}")

    # === Plotting historical and predicted Evaporation & Wind Speed ===
    df_hist_plot = df_hist[['Date', 'Evaporation', 'wind_speed']].dropna()
    df_hist_plot = df_hist_plot[df_hist_plot['Date'] <= last_date]

    fig = go.Figure()

    # Evaporation
    fig.add_trace(go.Scatter(x=df_hist_plot['Date'], y=df_hist_plot['Evaporation'],
                             mode='lines', name='Historical Evaporation'))
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Evaporation'],
                             mode='lines', name='Predicted Evaporation'))

    # Wind Speed
    fig.add_trace(go.Scatter(x=df_hist_plot['Date'], y=df_hist_plot['wind_speed'],
                             mode='lines', name='Historical Wind Speed'))
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['wind_speed'],
                             mode='lines', name='Predicted Wind Speed'))

    fig.update_layout(
        title=f'Evaporation & Wind Speed: Historical vs Forecast ({num_days} days)',
        xaxis_title='Date',
        yaxis_title='Evaporation-rate (mm/day)',
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white',
        height=600,
        width=1800
    )

    plot_html = fig.to_html(full_html=False)


    return df_future[['Date', 'Evaporation', 'Slope', 'Psychrometric constant',
                          'Radiation-Based_evaporation', 'Aerodynamics_evaporation',
                          'Saturation_vapour', 'Actual_vapour', 'Solar-Irradiance', 'wind_speed']], plot_html

# === Entry Point ===
if __name__ == "__main__":
    train_all_models()




import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.regularizers import l2
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import joblib, random
from datetime import timedelta

# Constants
A = 14.062
B = 44.722
C = 56.651
D = 34.955

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds(42)

# Helper: Fix date format from "01-06-2023 0.00" → "01-06-2023 00:00"
def fix_date_format(date_str):
    parts = re.split(r"[.\s]", str(date_str))
    if len(parts) >= 2:
        try:
            hour = int(float(parts[1]))
            return f"{parts[0]} {hour:02d}:00"
        except:
            return date_str
    return date_str

# Detect frequency: hourly, daily, monthly
def detect_frequency(df):
    diffs = df['Date'].sort_values().diff().dropna()
    avg_diff = diffs.mean()
    if avg_diff <= pd.Timedelta(hours=1):
        return 'hourly'
    elif avg_diff <= pd.Timedelta(days=1):
        return 'daily'
    else:
        return 'monthly'

# Normalize
def normalize_data(df_evap, df_wl):
    freq = detect_frequency(df_evap)

    if freq == 'hourly':
        df_evap = df_evap.set_index('Date').resample('D').sum().reset_index()
        df_wl = df_wl.set_index('Date').resample('D').mean().reset_index()
    elif freq == 'monthly':
        df_evap = df_evap.set_index('Date').resample('M').mean().reset_index()
        df_wl = df_wl.set_index('Date').resample('M').mean().reset_index()

    return df_evap, df_wl, freq

def normalize_data(df_evap, df_wl=None):
    freq = detect_frequency(df_evap)

    df_evap = df_evap.set_index('Date')

    if freq == 'hourly':
        df_evap = df_evap.resample('D').sum().reset_index()
        if df_wl is not None:
            df_wl = df_wl.set_index('Date').resample('D').mean().reset_index()
    elif freq == 'monthly':
        df_evap = df_evap.resample('M').mean().reset_index()
        if df_wl is not None:
            df_wl = df_wl.set_index('Date').resample('M').mean().reset_index()
    else:
        df_evap = df_evap.reset_index()
        if df_wl is not None:
            df_wl = df_wl.reset_index()

    if df_wl is not None:
        return df_evap, df_wl, freq
    return df_evap, freq

# Volume and concentration logic
def calculate_volume(wl):
    return A * wl**3 + B * wl**2 + C * wl + D

def calculate_surface_area(wl):
    return 3 * A * wl + 2 * B * wl + C

def calculate_concentration(wl_df, evap_df, in_con, mode='a', pond_name=None):
    if mode == 'a':
        # 🔹 ZONE MODE: Requires wl_df
        df = pd.merge(wl_df, evap_df, on='Date', how='inner')
        df = df.dropna(subset=['Wl', 'Evaporation'])

        df['Volume'] = np.where(
            df['Wl'] > 0.6,
            263.99 * df['Wl'] - 75.552,
            11.36 * df['Wl'] ** 6 + 52.102 * df['Wl'] ** 5 + 83.822 * df['Wl'] ** 4 +
            53.667 * df['Wl'] ** 3 + 19.7 * df['Wl'] ** 2 + 36.567 * df['Wl'] + 35.112
        )

        # Surface area calculation with reversed logic
        df['SurfaceArea'] = np.where(
            df['Wl'] > 0.6,
            263.99,
            68.16 * df['Wl'] ** 5 +
            260.51 * df['Wl'] ** 4 +
            335.288 * df['Wl'] ** 3 +
            161.001 * df['Wl'] ** 2 +
            39.4 * df['Wl'] +
            36.567
        )
        df['EvapVol'] = df['Evaporation'] * df['SurfaceArea'] / 1000  # mm * m² → m³
        df['V2'] = df['Volume'] - df['EvapVol']
        print(in_con)
        df['C2 (wt%)'] = (df['Volume'] * in_con) / df['V2']
        return df[['Date', 'Wl', 'Evaporation', 'Volume', 'V2', 'SurfaceArea', 'EvapVol', 'C2 (wt%)']]

    else:
        pond_volumes = {
            'A': 584994615.9,
            'B': 558427274.0,
            'C': 478725248.4,
            'D': 399030064.8,
            'E': 319476247.8,
            'F': 240507196.2
        }
        # 🔹 POND MODE: No wl_df needed
        df = evap_df.copy()
        df = df.dropna(subset=['Evaporation'])
        volume_m3 = pond_volumes.get(pond_name.upper(), 584994.6)  # Default to Pond A if not found
        df['Volume_m3'] = volume_m3 # fixed volume in m³
        print("pond name : ",pond_name)
        df['SurfaceArea'] = 9       # fixed surface area in m²
        df['Volume'] = df['Volume_m3']
        df['EvapVol'] = df['Evaporation']  * df['SurfaceArea'] / 1000
        df['V2'] = df['Volume'] - df['EvapVol']
        print(in_con)
        df['C2 (wt%)'] = (df['Volume'] * in_con) / df['V2']
        return df[['Date', 'Evaporation', 'Volume', 'V2', 'SurfaceArea', 'EvapVol', 'C2 (wt%)']]

# Plotting
def plot_concentration_with_forecast(forecast_df):
    # === Load historical C2 from saved file ===
    df_path = "outputs/bromine_concentration_output.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError("⚠ Historical C2 data file not found.")

    historical_df = pd.read_csv(df_path, parse_dates=["Date"])
    historical_df = historical_df[['Date', 'C2 (wt%)']].dropna()
    historical_df = historical_df[(historical_df['Date'] >= '2023-01-01') & (historical_df['Date'] <= '2023-12-31')]

    # === Plotting ===
    fig = go.Figure()

    # Historical Trace
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['C2 (wt%)'],
        mode='lines',
        name='Historical C2 (Jun–Dec 2023)',
        line=dict(color='blue')
    ))

    # Forecast Trace
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['C2 (wt%)'],
        mode='lines',
        name='Forecasted C2 (2024)',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Bromine Concentration",
        xaxis_title="Date",
        yaxis_title="C2 (wt%)",
        template="plotly_white",
        height=600,
        width=1800
    )

    return fig.to_html(full_html=False)

def plot_concentration(result_df):
    # === Plotting ===
    fig = go.Figure()

    # Forecast Trace
    fig.add_trace(go.Scatter(
        x=result_df['Date'],
        y=result_df['C2 (wt%)'],
        mode='lines',
        name='Forecasted C2 (2024)',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title="Bromine Concentration",
        xaxis_title="Date",
        yaxis_title="C2 (wt%)",
        template="plotly_white",
        height=600,
        width=1800
    )

    return fig.to_html(full_html=False)

def predict_future_concentration_xgb(user_start, user_end, mode='a', pond_name=None):
    lag_days = 120
    start_date = pd.to_datetime(user_start + "-01", format="%Y-%m-%d")
    end_date = pd.to_datetime(user_end + "-01", format="%Y-%m-%d") + pd.offsets.MonthEnd(0)
    forecast_days = (end_date - start_date).days + 1

    # === File paths ===
    if mode == 'b' and pond_name:
        pond_key = pond_name.upper()
        df_path = f"outputs/bromine_concentration_output_pond_{pond_key}.csv"
        model_path = f"models/xgb_c2_model_pond_{pond_key}.json"
        scaler_path = f"models/xgb_c2_scaler_pond_{pond_key}.save"
    else:
        df_path = "outputs/bromine_concentration_output.csv"
        model_path = "models/xgb_c2_model.json"
        scaler_path = "models/xgb_c2_scaler.save"

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"❌ Required C2 data file not found at: {df_path}")

    # === Load and preprocess ===
    df = pd.read_csv(df_path, parse_dates=["Date"])
    if mode == 'a':
        df = df[["Date", "C2 (wt%)", "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]].dropna()
        df['C2'] = df['C2 (wt%)']
    else:
        df = df[["Date", "C2 (wt%)", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]].dropna()
        df['C2'] = df['C2 (wt%)']

    df = df.set_index("Date").asfreq("D").interpolate(method='time')

    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["trend"] = np.arange(len(df))
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # === Rolling features ===
    rolling_windows = [14, 30, 60]
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["C2"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["C2"].rolling(window=window).std()
    df["cumulative_mean"] = df["C2"].expanding().mean()
    df["cumulative_std"] = df["C2"].expanding().std()

    # === Lag features ===
    for i in range(1, lag_days + 1):
        df[f"lag_{i}"] = df["C2"].shift(i)

    df.dropna(inplace=True)

    # === Features ===
    feature_cols = ["dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                       "cumulative_mean", "cumulative_std",
                       "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"
                   ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                         rolling_windows] \
                   + [f"lag_{i}" for i in range(1, lag_days + 1)] if mode == 'a' else [
                       "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
                       "cumulative_mean", "cumulative_std",
                       "Evaporation", "Volume", "SurfaceArea", "EvapVol"
                   ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in
                                                                         rolling_windows] \
                   + [f"lag_{i}" for i in range(1, lag_days + 1)]

    X = df[feature_cols].values
    y = df["C2"].values

    # === Scale ===
    os.makedirs("models", exist_ok=True)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        try:
            scaler.transform(X)
        except ValueError:
            print("⚠️ Feature mismatch. Re-fitting scaler.")
            scaler = MinMaxScaler()
            scaler.fit(X)
            joblib.dump(scaler, scaler_path)
            if os.path.exists(model_path):
                os.remove(model_path)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        joblib.dump(scaler, scaler_path)

    X_scaled = scaler.transform(X)

    # === Train XGBoost ===
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9)
    model.fit(X_scaled, y)

    # === Forecasting ===
    last_known_date = df.index[-1]
    past_c2 = df["C2"].iloc[-lag_days:].tolist()
    series_dict = {col: df[col] for col in ["Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol"]} if mode == 'a' else {
        col: df[col] for col in ["Evaporation", "Volume", "SurfaceArea", "EvapVol"]}

    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_c2 = []

    for i, next_date in enumerate(future_dates):
        dayofyear = next_date.dayofyear
        month = next_date.month
        trend = len(df) + i - 1
        dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365)
        dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        rolling_means = [np.mean(past_c2[-w:]) for w in rolling_windows]
        rolling_stds = [np.std(past_c2[-w:]) for w in rolling_windows]
        cumulative_mean = np.mean(past_c2)
        cumulative_std = np.std(past_c2)

        def get_series_val(s, idx):
            try:
                return s.loc[next_date]
            except KeyError:
                return s.iloc[idx % len(s)]

        if mode == 'a':
            wl = get_series_val(series_dict["Wl"], i)
        evaporation = get_series_val(series_dict["Evaporation"], i)
        volume = get_series_val(series_dict["Volume"], i)
        surface_area = get_series_val(series_dict["SurfaceArea"], i)
        evapvol = get_series_val(series_dict["EvapVol"], i)
        lag_features = past_c2[-lag_days:]

        raw_input_row = np.array([[dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                                   cumulative_mean, cumulative_std,
                                   wl, evaporation, volume, surface_area, evapvol] +
                                  rolling_means + rolling_stds + lag_features]) if mode == 'a' else np.array([
            [dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend, cumulative_mean, cumulative_std,
                evaporation, volume, surface_area, evapvol] + rolling_means + rolling_stds + lag_features])

        if raw_input_row.shape[1] != scaler.n_features_in_:
            raise ValueError(f"❌ Feature count mismatch: {raw_input_row.shape[1]} vs expected {scaler.n_features_in_}")

        input_scaled = scaler.transform(raw_input_row)
        pred = model.predict(input_scaled)[0]

        future_c2.append(pred)
        past_c2.append(pred)

    forecast_df = pd.DataFrame({"Date": future_dates, "C2 (wt%)": future_c2})

    return forecast_df, df.reset_index()

def run_bromine_concentration(wl_path, evap_path, in_con, mode, pond_name=None, forecast=True, user_start=None, user_end=None):
    print(f"⚙ Mode selected: {mode}")
    try:
        # Pre-check frequency using a light read
        temp_evap = pd.read_csv(evap_path, usecols=['Date'])
        temp_evap['Date'] = temp_evap['Date'].apply(fix_date_format)
        temp_evap['Date'] = pd.to_datetime(temp_evap['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
        is_hourly = detect_frequency(temp_evap) == 'hourly'
    except Exception as e:
        print("⚠ Frequency detection failed:", e)
        is_hourly = False

    try:
        if mode == 'a':
            # === ZONE MODE ===
            if is_hourly:
                wl_df = pd.read_csv(wl_path)
                evap_df = pd.read_csv(evap_path)
                wl_df['Date'] = wl_df['Date'].apply(fix_date_format)
                evap_df['Date'] = evap_df['Date'].apply(fix_date_format)
                wl_df['Date'] = pd.to_datetime(wl_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
                evap_df['Date'] = pd.to_datetime(evap_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
            else:
                wl_df = pd.read_csv(wl_path, parse_dates=['Date'], dayfirst=True)
                evap_df = pd.read_csv(evap_path, parse_dates=['Date'], dayfirst=True)

            evap_df, wl_df, freq = normalize_data(evap_df, wl_df)
            result_df = calculate_concentration(wl_df, evap_df, in_con, mode=mode)

            # 🟦 Save calculated C2
            output_path = os.path.join("outputs", "bromine_concentration_output.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 GRU Forecast only for ZONE
            if forecast:
                forecast_df, _ = predict_future_concentration_xgb(user_start=user_start, user_end=user_end)
                forecast_df.to_csv("outputs/forecasted_c2.csv", index=False)
                plot_html = plot_concentration_with_forecast(forecast_df)

                return forecast_df, plot_html

            # If no forecast requested
            plot_html = plot_concentration(result_df)

            figV = go.Figure()
            figV.add_trace(go.Scatter(x=result_df['Date'],y=result_df['Volume'],mode='lines',name='Volume V1(2024)',line=dict(color='red')))
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['V2'], mode='lines', name='Volume V2(2024)',line=dict(color='green')))
            figV.update_layout(title="Volume Visualization",xaxis_title="Date",yaxis_title="Volume (v1 & v2)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white",height=600,width=1800)
            plot_v = figV.to_html(full_html=False)

            return result_df, plot_html, plot_v

        else:
            # === POND MODE === (No GRU prediction)
            if is_hourly:
                evap_df = pd.read_csv(evap_path)
                evap_df['Date'] = evap_df['Date'].apply(fix_date_format)
                evap_df['Date'] = pd.to_datetime(evap_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
            else:
                evap_df = pd.read_csv(evap_path, parse_dates=['Date'], dayfirst=True)

            evap_df, freq = normalize_data(evap_df)
            result_df = calculate_concentration(None, evap_df, in_con, mode=mode, pond_name=pond_name)

            # Save result for continuity if needed
            output_path = os.path.join("outputs", f"bromine_concentration_output_pond_{pond_name}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 Forecast (same XGBoost model)
            if forecast:
                forecast_df, _ = predict_future_concentration_xgb(mode='b',pond_name=pond_name, user_start=user_start, user_end=user_end)
                forecast_df.to_csv(f"outputs/forecasted_c2_pond_{pond_name}.csv", index=False)
                plot_html = plot_concentration(forecast_df)
                figV = go.Figure()
                figV.add_trace(
                    go.Scatter(x=result_df['Date'], y=result_df['Volume'], mode='lines', name='Volume V1(2024)',
                               line=dict(color='red')))
                figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['V2'], mode='lines', name='Volume V2(2024)',
                                          line=dict(color='green')))
                figV.update_layout(title="Volume Visualization", xaxis_title="Date", yaxis_title="Volume (v1 & v2)",
                                   legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                                   template="plotly_white", height=600, width=1800)
                plot_v = figV.to_html(full_html=False)

                return forecast_df, plot_html, plot_v

            plot_html = plot_concentration(result_df)

            figV = go.Figure()
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Volume'], mode='lines', name='Volume V1(2024)',line=dict(color='red')))
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['V2'], mode='lines', name='Volume V2(2024)',line=dict(color='green')))
            figV.update_layout(title="Volume Visualization", xaxis_title="Date",yaxis_title="Volume (v1 & v2)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white", height=600, width=1800)
            plot_v = figV.to_html(full_html=False)

            return result_df, plot_html, plot_v

    except Exception as e:
        print("❌ Prediction failed:", e)
        raise

if __name__ == "__main__":
    # predict_future_concentration_xgb(days=365)
    predict_future_concentration_xgb(user_start='2025-01',user_end='2025-12',mode='b', pond_name='A')
