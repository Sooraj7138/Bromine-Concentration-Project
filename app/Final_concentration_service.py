import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime

# === Constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'predictions_now.csv')

TARGET = 'Final_Concentration_(gpl)'
FEATURES = ['Brine_In', 'Brine_Out', 'Initial_Concentration_(gpl)', 
            'Day', 'Month', 'Year', 'Month_sin', 'Month_cos']

# === Helper Function ===
def preprocess_dates(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().any():
        raise ValueError("❌ Some values in 'Date' column could not be parsed to datetime.")

    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    return df

# === Train Model ===
def train_C2_model(pond_name):
    # Determine data path based on pond name
    if pond_name == "Dhordo":
        data_path = os.path.join(BASE_DIR, 'data', 'test.csv')
    else:
        # Defaulting to Khavda for other pond names or if not specified
        data_path = os.path.join(BASE_DIR, 'data', 'test_khavda.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Training data not found at {data_path}")

    print(f"🔧 Training model for pond: {pond_name} using {data_path}")
    df = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = preprocess_dates(df)

    # Features = your original parameters + date-derived features
    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_filename = f'bromine_model_{pond_name.lower()}.joblib'
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)
    
    print(f"✅ Bromine model for {pond_name} trained successfully.")
    return model, FEATURES

# === Forecast Function ===
def forecast_future_C2(start_date_str, end_date_str, c2_path, pond_name):
    # 1. Load the pre-trained model based on pond_name
    model_filename = f'bromine_model_{pond_name.lower()}.joblib'
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model for {pond_name} not found at {model_path}. Training it now...")
        model, features = train_C2_model(pond_name)
    else:
        print(f"✅ Loading existing model for {pond_name} from {model_path}")
        model = joblib.load(model_path)
        features = FEATURES

    # 2. Load the input file (c2_path)
    if not os.path.exists(c2_path):
        raise FileNotFoundError(f"❌ Input file not found at {c2_path}")
    
    df_new = pd.read_csv(c2_path, parse_dates=['Date'], dayfirst=True)
    df_new = preprocess_dates(df_new)

    # 3. Filter by date range
    # Assuming start_date_str and end_date_str are in 'YYYY-MM' format
    start_date = pd.to_datetime(start_date_str + "-01", format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date_str + "-01", format="%Y-%m-%d") + pd.offsets.MonthEnd(0)
    
    mask = (df_new['Date'] >= start_date) & (df_new['Date'] <= end_date)
    df_filtered = df_new.loc[mask].copy()

    if df_filtered.empty:
        print(f"⚠️ No data found in the range {start_date_str} to {end_date_str}. Using all available data from input file.")
        df_filtered = df_new.copy()

    # 4. Predict
    X_new = df_filtered[features]
    df_filtered["predicted_final"] = model.predict(X_new)

    # 5. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_PATH}")

    # 6. Generate Plotly Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered["predicted_final"],
                             mode='lines', name=f'Predicted Concentration ({pond_name})'))

    fig.update_layout(
        title=f'Predicted Final Concentration for {pond_name}',
        xaxis_title='Date',
        yaxis_title='Final Concentration (gpl)',
        template='plotly_white',
        height=600,
        width=1800
    )

    plot_html = fig.to_html(full_html=False)
    
    # Rename column for compatibility with route expectation
    df_filtered['Final Concentration (gpl)'] = df_filtered['predicted_final']
    
    return df_filtered[['Date', 'Final Concentration (gpl)']], plot_html

if __name__ == "__main__":
    # Example usage for testing
    # forecast_future_C2('2025-01', '2025-12', os.path.join(BASE_DIR, 'data', 'tester.csv'), "Khavda")
    pass
