import pandas as pd
from prophet import Prophet
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

CSV_PATH = r"C:\Users\Computer Hub\Desktop\gold-predictor\XAU_USD Historical Data.csv"
MODEL_PATH = "model_2024_train.joblib"
HISTORY_PATH = "history_2024_train.csv"
FORECAST_PATH = "forecast_2025.csv"

def train_and_forecast():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV not found")

    # ------------------- Load CSV -------------------
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]

    # Detect price column
    price_col = next((c for c in df.columns if 'price' in c.lower() or 'close' in c.lower()), None)
    df[price_col] = df[price_col].astype(str).str.replace(',', '').str.replace('%', '')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    # Detect date column
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()), None)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)

    # ------------------- Use data up to 2024-12-31 -------------------
    df_train = df[df[date_col] <= '2024-12-31']
    clean = pd.DataFrame({'ds': df_train[date_col], 'y': df_train[price_col]})

    # ------------------- Train Prophet Model -------------------
    model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_prior_scale=10,
        changepoint_range=0.95
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(clean)

    # ------------------- Evaluate on Historical Data (up to 2024) -------------------
    forecast_in_sample = model.predict(clean[['ds']])
    y_true = clean['y'].values
    y_pred = forecast_in_sample['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("✅ Model Accuracy on Historical Data (up to 2024):")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # ------------------- Forecast 2025 (e.g., 1-2 months) -------------------
    future = model.make_future_dataframe(periods=60, freq='D')  # 2 months
    forecast_2025 = model.predict(future)

    # ------------------- Save model, history, forecast -------------------
    dump(model, MODEL_PATH)
    clean.to_csv(HISTORY_PATH, index=False)
    forecast_2025.to_csv(FORECAST_PATH, index=False)

    print(f"✅ Forecast for 2025 saved to {FORECAST_PATH}")
    print(f"✅ Model trained on data up to 2024 saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_forecast()
