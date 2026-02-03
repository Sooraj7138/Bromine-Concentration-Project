import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# === Constants ===
DATA_PATH = r"D:\Sooraj\Project_Bromine_Concentration/app\data/test.csv"
TEST_PATH = r"D:\Sooraj\Project_Bromine_Concentration/app\data/tester.csv"
OUTPUT_PATH = r"D:\Sooraj\Project_Bromine_Concentration/app/outputs\predictions_now.csv"

TARGET = 'Final_Concentration_(gpl)'

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
def train_bromine_model():
    # Load and preprocess training data
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = preprocess_dates(df)

    # Features = your original parameters + date-derived features
    FEATURES = ['Brine_In', 'Brine_Out', 'Initial_Concentration_(gpl)',
                'Day', 'Month', 'Year', 'Month_sin', 'Month_cos']

    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost model (same params as waterlevel script)
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

    print("✅ Bromine model trained successfully.")
    return model, FEATURES

# === Predict on new data ===
def predict_new_data(model, FEATURES):
    new_data = pd.read_csv(TEST_PATH, parse_dates=['Date'], dayfirst=True)
    new_data = preprocess_dates(new_data)

    # Use the same features as training
    X_new = new_data[FEATURES]
    new_data["predicted_final"] = model.predict(X_new)

    output_cols = ['Date', 'Brine_In', 'Brine_Out', 'Initial_Concentration_(gpl)', 'predicted_final']
    new_data[output_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    model, FEATURES = train_bromine_model()
    predict_new_data(model, FEATURES)