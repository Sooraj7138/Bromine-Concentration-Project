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

def calculate_concentration(wl_df, evap_df, in_df, out_df, in_con, mode='a', pond_name=None):
    if mode == 'a':
        # Merge and clean
        df = pd.merge(wl_df, evap_df, on='Date', how='inner')
        df = df.dropna(subset=['Wl', 'Evaporation'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Prepare lists
        volumes, surface_areas, evap_vols, v2s = [], [], [], []
        initial_concs, final_concs, quantities = [], [], []

        current_conc = in_con  # Starting concentration
        prev_quantity = None
        prev_v2 = None

        # Loop through days
        for i in range(len(df)):
            wl = df.loc[i, 'Wl']
            evap = df.loc[i, 'Evaporation']

            # Volume & Surface Area
            if wl > 0.6:
                volume = 263.99 * wl - 75.552
                surface_area = 263.99
            else:
                volume = (
                    11.36 * wl ** 6 +
                    52.102 * wl ** 5 +
                    83.822 * wl ** 4 +
                    53.667 * wl ** 3 +
                    19.7 * wl ** 2 +
                    36.567 * wl +
                    35.112
                )
                surface_area = (
                    68.16 * wl ** 5 +
                    260.51 * wl ** 4 +
                    335.288 * wl ** 3 +
                    161.001 * wl ** 2 +
                    39.4 * wl +
                    36.567
                )

            volume_m3 = volume * 1000000
            evap_vol = evap * surface_area / 1000  # mm to m³
            v2_kg = volume - evap_vol
            v2 = v2_kg * 1000000
            final_conc = (volume * current_conc) / v2_kg

            # Quantity in tonnes
            if i == 0:
                quantity_kg = v2 * current_conc
                quantity = quantity_kg / 1000
            else:
                quantity_kg = prev_quantity + (v2 - prev_v2) * in_con
                quantity = quantity_kg / 1000

            # Store
            volumes.append(volume_m3)
            surface_areas.append(surface_area)
            evap_vols.append(evap_vol)
            v2s.append(v2)
            initial_concs.append(round(current_conc, 3))
            final_concs.append(round(final_conc, 3))
            quantities.append(round(quantity, 3))

            # Update for next loop
            current_conc = final_conc
            prev_quantity = quantity_kg
            prev_v2 = v2

        # Assign to DataFrame
        df['Water Level (m)'] = df['Wl']
        df['Evaporation (mm/day)'] = df['Evaporation']
        df['Volume V1(m3)'] = volumes
        df['Surface Area (mm2)'] = surface_areas
        df['Evaporation Volume (mm3)'] = evap_vols
        df['Volume V2(m3)'] = v2s
        df['Initial Concentration C1 (gpl)'] = initial_concs
        df['Final Concentration C2 (gpl)'] = final_concs
        df['Quantity (tonnes)'] = quantities

        return df[['Date', 'Water Level (m)', 'Evaporation (mm/day)', 'Volume V1(m3)', 'Evaporation Volume (mm3)', 'Surface Area (mm2)', 'Volume V2(m3)',
                   'Initial Concentration C1 (gpl)', 'Final Concentration C2 (gpl)', 'Quantity (tonnes)']]

    else:
        # pond_volumes = {
        #     'A': 584994615.9,
        #     'B': 558427274.0,
        #     'C': 478725248.4,
        #     'D': 399030064.8,
        #     'E': 319476247.8,
        #     'F': 240507196.2
        # }
        #
        #
        # pondNames = ['C4', 'C5', 'C6', 'C10']
        # for i in range(11, 51):
        #     names = f"C{i}"
        #     pondNames.append(names)
        #
        # condenser_data = {
        #     'C4': 54.6, 'C5': 72.8, 'C6': 61.1, 'C10': 16.4,
        #     'C11': 258.2, 'C12': 92.3, 'C13': 66.2, 'C14': 52.2, 'C15': 67.7,
        #     'C16': 71.8, 'C17': 80.4, 'C18': 95.2, 'C19': 74.9, 'C20': 76.1,
        #     'C21': 125, 'C22': 125, 'C23': 110, 'C24': 173, 'C25': 173,
        #     'C26': 110, 'C27': 110, 'C28': 110, 'C29': 220, 'C30': 116,
        #     'C31': 110, 'C32': 100, 'C33': 125, 'C34': 125, 'C35': 180,
        #     'C36': 250, 'C37': 350, 'C38': 350, 'C39': 350, 'C40': 350,
        #     'C41': 285, 'C42': 260, 'C43': 25, 'C44': 230, 'C45': 240,
        #     'C46': 300, 'C47': 300, 'C48': 300, 'C49': 625, 'C50': 325
        # }
        #
        # # Multiply each area by 10,000 and store in a list
        # pondArea_m2 = [area * 10000 for area in condenser_data.values()]
        #
        # for i in range(len(pondNames)):
        #     pondNames[i] = pondArea_m2[i]
        #
        # print(pondNames[3])
        # print(pondArea_m2)

        # 🔹 POND MODE: No wl_df needed
        evap_df_value = evap_df.copy()
        in_df_value = in_df.copy()
        out_df_value = out_df.copy()

        for df in [evap_df_value, in_df_value, out_df_value]:
            df['Date'] = pd.to_datetime(df['Date'])
        df = evap_df.merge(in_df_value, on='Date').merge(out_df_value, on='Date')
        df = df.dropna(subset=['Evaporation', 'Brine_In', 'Brine_Out'])
        # df['Evaporation'] = df['Evaporation'].cumsum()

        #cumulative difference of Brine
        df['NetFlow'] = df['Brine_In'] - df['Brine_Out']
        df['CumulativeNetFlow'] = df['NetFlow'].cumsum()

        # Prepare lists to store results
        volumes, surface_areas, brine_ins_outs, cumulative_brine_flows, water_levels, v2s = [], [], [], [], [], []
        initial_concs, final_concs, added_const_wls, evap_losses, es, bromide_qtys, evaps = [], [], [], [], [], [], []

        # Set the starting concentration
        current_conc = in_con

        # Loop through each day
        for i in range(len(df)):
            evap = df.loc[i, 'Evaporation']
            cumulative_brine_flow = df.loc[i, 'CumulativeNetFlow']
            surface_area = 77363631.45 if pond_name == "dhordo" else 84279908.54  # fixed surface area in m²

            water_level = cumulative_brine_flow / surface_area
            added_const_wl = 0.8 + water_level
            volume = surface_area * added_const_wl

            evap_loss = (surface_area * (evap / 1000))
            # e = surface_area * evap_loss
            v2 = volume - evap_loss

            final_conc = (volume * current_conc) / v2
            if i == 0:
                bromide_qty = current_conc * v2
            else:
                bromide_qty = previous_bromide_qty + (v2 - previous_v2) * in_con
            bromide_kg = bromide_qty / 1000

            # Store for DataFrame
            volumes.append(round(volume,2))
            surface_areas.append(round(surface_area,2))
            cumulative_brine_flows.append(cumulative_brine_flow)
            v2s.append(round(v2,2))
            water_levels.append(round(water_level,6))
            evaps.append(evap)
            added_const_wls.append(round(added_const_wl,6))
            evap_losses.append(round(evap_loss,2))
            # es.append(e)
            initial_concs.append(round(current_conc,3))
            final_concs.append(round(final_conc,3))
            bromide_qtys.append(round(bromide_kg))

            # Update current_conc for next day
            current_conc = final_conc

            previous_bromide_qty = bromide_qty
            previous_v2 = v2

        # Assign columns to DataFrame
        df['Brine_Differenced'] = cumulative_brine_flows
        df['SurfaceArea (m2)'] = surface_areas
        df['Water Level (m)'] = added_const_wls
        df['Volume V1 (m3)'] = volumes
        df['Evaporation (mm/day)'] = evaps
        df['Evaporation Volume (mm3)'] = evap_losses
        # df['e_(m3)'] = es
        df['Volume V2 (m3)'] = v2s
        df['Initial Concentration (gpl)'] = initial_concs
        df['Final Concentration C2 (gpl)'] = final_concs
        df['Quantity (tonnes)'] = bromide_qtys

        print("pond name : ",pond_name)

        return df[['Date', 'Water Level (m)', 'Volume V1 (m3)', 'Evaporation (mm/day)', 'Evaporation Volume (mm3)', 'SurfaceArea (m2)', 'Volume V2 (m3)', 'Initial Concentration (gpl)', 'Final Concentration C2 (gpl)', 'Quantity (tonnes)']]

# Plotting
def plot_concentration_with_forecast(forecast_df):
    # === Load historical C2 from saved file ===
    df_path = "outputs/bromine_concentration_output.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError("⚠ Historical C2 data file not found.")


    historical_df = pd.read_csv(df_path, parse_dates=["Date"])
    historical_df = historical_df[['Date', 'C2 (mg/l)']].dropna()
    historical_df = historical_df[(historical_df['Date'] >= '2023-01-01') & (historical_df['Date'] <= '2023-12-31')]

    # === Plotting ===
    fig = go.Figure()

    # Historical Trace
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['C2 (mg/l)'],
        mode='lines',
        name='Historical Final Bromine Concentration  (Jun–Dec 2023)',
        line=dict(color='blue')
    ))

    # Forecast Trace
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['C2 (mg/l)'],
        mode='lines',
        name='Forecasted Final Bromine Concentration',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Bromine Concentration",
        xaxis_title="Date",
        yaxis_title="Final Bromine Concentration (gpl)",
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
        y=result_df['Final Concentration C2 (gpl)'],
        mode='lines',
        name='Final Bromine Concentration',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title="Final Bromine Concentration (gpl)",
        xaxis_title="Date",
        yaxis_title="Final Bromine Concentration (gpl)",
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
        df = df[["Date", "C2 (mg/l)", "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"]].dropna()
        df['C2'] = df['C2 (mg/l)']
    else:
        df = df[["Date", "C2 (mg/l)", "Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"]].dropna()
        df['C2'] = df['C2 (mg/l)']

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

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Debug print to check the shape and head of the DataFrame
    print("DataFrame shape:", df.shape)
    print("DataFrame head:\n", df.head())

    # === Features ===
    feature_cols = [
        "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
        "cumulative_mean", "cumulative_std",
        "Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"
    ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in rolling_windows] + [
        f"lag_{i}" for i in range(1, lag_days + 1)
    ] if mode == 'a' else [
        "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "trend",
        "cumulative_mean", "cumulative_std",
        "Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"
    ] + [f"rolling_mean_{w}" for w in rolling_windows] + [f"rolling_std_{w}" for w in rolling_windows] + [
        f"lag_{i}" for i in range(1, lag_days + 1)
    ]

    X = df[feature_cols].values
    y = df["C2"].values

    if X.size == 0:
        print("❌ No valid data available for the selected date range. Prediction aborted.")
        return pd.DataFrame(), None

    # Debug print to check the shape of X and feature columns
    print("Shape of X:", X.shape)
    print("Feature columns:", feature_cols)

    # === Scale ===
    os.makedirs("models", exist_ok=True)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        try:
            X_scaled = scaler.transform(X)
        except ValueError:
            print("⚠️ Feature mismatch. Re-fitting scaler.")
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)
            if os.path.exists(model_path):
                os.remove(model_path)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)

    # === Train XGBoost ===
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9)
    model.fit(X_scaled, y)

    # === Forecasting ===
    last_known_date = df.index[-1]
    past_c2 = df["C2"].iloc[-lag_days:].tolist()

    series_dict = {
        col: df[col] for col in ["Wl", "Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"]
    } if mode == 'a' else {
        col: df[col] for col in ["Evaporation", "Volume", "SurfaceArea", "EvapVol", "Initial_Conc"]
    }

    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_c2 = []

    for i, next_date in enumerate(future_dates):
        dayofyear = next_date.dayofyear
        month = next_date.month
        trend = len(df) + i
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
            raw_input_row = [dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                             cumulative_mean, cumulative_std, wl,
                             get_series_val(series_dict["Evaporation"], i),
                             get_series_val(series_dict["Volume"], i),
                             get_series_val(series_dict["SurfaceArea"], i),
                             get_series_val(series_dict["EvapVol"], i),
                             get_series_val(series_dict["Initial_Conc"], i)]
        else:
            raw_input_row = [dayofyear_sin, dayofyear_cos, month_sin, month_cos, trend,
                             cumulative_mean, cumulative_std,
                             get_series_val(series_dict["Evaporation"], i),
                             get_series_val(series_dict["Volume"], i),
                             get_series_val(series_dict["SurfaceArea"], i),
                             get_series_val(series_dict["EvapVol"], i),
                             get_series_val(series_dict["Initial_Conc"], i)]

        raw_input_row += rolling_means + rolling_stds + past_c2[-lag_days:]

        if len(raw_input_row) != scaler.n_features_in_:
            raise ValueError(f"❌ Feature count mismatch: {len(raw_input_row)} vs expected {scaler.n_features_in_}")

        input_scaled = scaler.transform([raw_input_row])
        pred = model.predict(input_scaled)[0]
        future_c2.append(pred)
        past_c2.append(pred)

    forecast_df = pd.DataFrame({"Date": future_dates, "C2 (mg/l)": future_c2})
    return forecast_df, df.reset_index()

def run_bromine_concentration(wl_path, evap_path, in_path, out_path, in_con, mode, pond_name=None):
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
            result_df = calculate_concentration(wl_df, evap_df, None, None, in_con, mode=mode)

            # 🟦 Save calculated C2
            output_path = os.path.join("outputs", "bromine_concentration_output.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 GRU Forecast only for ZONE
            # if forecast:
            #     forecast_df, _ = predict_future_concentration_xgb(user_start=user_start, user_end=user_end, mode=mode, pond_name=pond_name)
            #     forecast_df.to_csv("outputs/forecasted_c2.csv", index=False)
            #     plot_html = plot_concentration_with_forecast(forecast_df)
            #
            #     figV = go.Figure()
            #     figV.add_trace(
            #         go.Scatter(x=result_df['Date'], y=result_df['Volume'], mode='lines', name='Volume V1(2024)',
            #                    line=dict(color='red')))
            #     figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['V2'], mode='lines', name='Volume V2(2024)',
            #                               line=dict(color='green')))
            #     figV.update_layout(title="Volume Visualization", xaxis_title="Date", yaxis_title="Volume (v1 & v2)",
            #                        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
            #                        template="plotly_white", height=600, width=1800)
            #     plot_v = figV.to_html(full_html=False)
            #
            #     figT = go.Figure()
            #     figT.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Quantity_tonnes'], mode='lines',
            #                               name='Quantity(2024)', line=dict(color='blue')))
            #     figT.update_layout(title="Quantity in tonnes Visualization", xaxis_title="Date", yaxis_title="Quantity in tonnes",
            #                        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
            #                        template="plotly_white", height=600, width=1800)
            #     plot_t = figT.to_html(full_html=False)
            #
            #     return forecast_df, plot_html, plot_v , plot_t

            # If no forecast requested
            plot_html = plot_concentration(result_df)

            figV = go.Figure()
            figV.add_trace(go.Scatter(x=result_df['Date'],y=result_df['Volume V1(m3)'],mode='lines',name='Volume V1',line=dict(color='red')))
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Volume V2(m3)'], mode='lines', name='Volume V2',line=dict(color='green')))
            figV.update_layout(title="Difference in Volume (V1 and V2) (m3)",xaxis_title="Date",yaxis_title="Volume (v1 & v2) (m3)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white",height=600,width=1800)
            plot_v = figV.to_html(full_html=False)

            figT = go.Figure()
            figT.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Quantity (tonnes)'], mode='lines',
                                      name='Quantity', line=dict(color='blue')))
            figT.update_layout(title="Quantity (tonnes)", xaxis_title="Date",
                               yaxis_title="Quantity (tonnes)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white", height=600, width=1800)
            plot_t = figT.to_html(full_html=False)

            return result_df, plot_html, plot_v, plot_t

        else:
            # === POND MODE === (No GRU prediction)
            if is_hourly:
                evap_df = pd.read_csv(evap_path)
                evap_df['Date'] = evap_df['Date'].apply(fix_date_format)
                evap_df['Date'] = pd.to_datetime(evap_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
            else:
                evap_df = pd.read_csv(evap_path, parse_dates=['Date'], dayfirst=True)
                in_df = pd.read_csv(in_path, parse_dates=['Date'], dayfirst=True)
                out_df = pd.read_csv(out_path, parse_dates=['Date'], dayfirst=True)
            evap_df, freq = normalize_data(evap_df)
            result_df = calculate_concentration(None, evap_df, in_df, out_df, in_con, mode=mode, pond_name=pond_name)

            # Save result for continuity if needed
            output_path = os.path.join("outputs", f"bromine_concentration_output.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)

            # 🔮 Forecast (same XGBoost model)
            # if forecast:
            #     forecast_df, _ = predict_future_concentration_xgb(user_start=user_start, user_end=user_end, mode='b',pond_name=pond_name)
            #     forecast_df.to_csv(f"outputs/forecasted_c2_pond_{pond_name}.csv", index=False)
            #     plot_html = plot_concentration(forecast_df)
            #     figV = go.Figure()
            #     figV.add_trace(
            #         go.Scatter(x=result_df['Date'], y=result_df['Volume'], mode='lines', name='Volume V1(2024)',
            #                    line=dict(color='red')))
            #     figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['V2'], mode='lines', name='Volume V2(2024)',
            #                               line=dict(color='green')))
            #     figV.update_layout(title="Volume Visualization", xaxis_title="Date", yaxis_title="Volume (v1 & v2)",
            #                        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
            #                        template=" plotly_white", height=600, width=1800)
            #     plot_v = figV.to_html(full_html=False)
            #
            #     return forecast_df, plot_html, plot_v

            plot_html = plot_concentration(result_df)

            figV = go.Figure()
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Volume V1 (m3)'], mode='lines', name='Volume V1',line=dict(color='red')))
            figV.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Volume V2 (m3)'], mode='lines', name='Volume V2',line=dict(color='green')))
            figV.update_layout(title="Difference in Volume (V1 and V2) (m3)", xaxis_title="Date",yaxis_title="Volume (v1 & v2) (m3)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white", height=600, width=1800)
            plot_v = figV.to_html(full_html=False)

            figT = go.Figure()
            figT.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Quantity (tonnes)'], mode='lines',
                                      name='Quantity', line=dict(color='blue')))
            figT.update_layout(title="Quantity (tonnes)", xaxis_title="Date",
                               yaxis_title="Quantity (tonnes)",
                               legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.4),
                               template="plotly_white", height=600, width=1800)
            plot_t = figT.to_html(full_html=False)

            return result_df, plot_html, plot_v, plot_t

    except Exception as e:
        print("❌ Prediction failed:", e)
        raise

# if __name__ == "__main__":
    # predict_future_concentration_xgb(days=365)
    # predict_future_concentration_xgb(user_start='2026-01',user_end='2026-12', mode='a')