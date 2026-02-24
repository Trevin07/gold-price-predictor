from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
import pandas as pd
import os, uuid
import plotly.graph_objs as go
import plotly.io as pio

# -------------------- Paths --------------------
MODEL_PATH = "model_2024_train.joblib"
HISTORY_PATH = "history_2024_train.csv"
FULL_CSV_PATH = r"C:\Users\Computer Hub\Desktop\gold-predictor\XAU_USD Historical Data.csv"
FORECAST_DIR = "forecasts"
os.makedirs(FORECAST_DIR, exist_ok=True)

# -------------------- FastAPI Setup --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------- Load Model --------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    else:
        raise HTTPException(status_code=500, detail="No trained model found.")

# -------------------- Plot Function --------------------
def create_plot(history, forecast, actual=None):
    # History line (up to 2024)
    p1 = go.Scatter(
        x=history['ds'], y=history['y'],
        mode='lines', name='History (up to 2024)',
        line=dict(color='blue', width=2)
    )
    
    # Forecast line (2025)
    p2 = go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast (2025)',
        line=dict(color='red', width=2)
    )
    
    data = [p1, p2]
    
    # Confidence interval
    p3 = go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    )
    p4 = go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    )
    data.extend([p3, p4])
    
    # Actual 2025 line (green) aligned with forecast dates
    if actual is not None and not actual.empty:
        forecast_dates = pd.to_datetime(forecast['ds'])
        actual_filtered = actual[actual['ds'].isin(forecast_dates)]
        if not actual_filtered.empty:
            p5 = go.Scatter(
                x=actual_filtered['ds'], y=actual_filtered['y'],
                mode='lines', name='Actual 2025',
                line=dict(color='green', width=2, dash='dot')  # dotted green line
            )
            data.append(p5)
    
    fig = go.Figure(data=data)
    
    fig.update_layout(
        title='Gold Price History + Forecast (2025)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        legend=dict(orientation='h', y=-0.2),
        hovermode='x unified'
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/forecast")
async def forecast_endpoint(horizon: int = Form(...), request: Request = None):
    try:
        # Load history up to 2024
        history = pd.read_csv(HISTORY_PATH, parse_dates=['ds'])
        model = load_model()

        # Forecast future
        future = model.make_future_dataframe(periods=horizon, freq='D')
        forecast = model.predict(future)

        # Filter only future forecast rows
        last_date = history['ds'].max()
        forecast_future = forecast[forecast['ds'] > last_date][['ds','yhat','yhat_lower','yhat_upper']].copy()
        forecast_future['ds'] = forecast_future['ds'].dt.date.astype(str)

        # Load full CSV and filter actuals for forecast period
        full_df = pd.read_csv(FULL_CSV_PATH, parse_dates=['Date'])
        price_col = next((c for c in full_df.columns if 'price' in c.lower() or 'close' in c.lower()), None)
        if price_col is None:
            raise ValueError(f"No price column found in full CSV. Columns: {list(full_df.columns)}")
        full_df[price_col] = pd.to_numeric(full_df[price_col].astype(str).str.replace(',', ''), errors='coerce')
        # Keep only actuals for forecast dates
        forecast_dates = pd.to_datetime(forecast_future['ds'])
        actual_df = full_df[full_df['Date'].isin(forecast_dates)][['Date', price_col]].rename(columns={'Date':'ds', price_col:'y'})

        # Create plot
        plot_div = create_plot(history, forecast_future, actual=actual_df)

        # Save forecast to Excel
        filename = str(uuid.uuid4()) + ".xlsx"
        excel_path = os.path.join(FORECAST_DIR, filename)
        forecast_future.to_excel(excel_path, index=False)

        return templates.TemplateResponse("results.html", {
            "request": request,
            "plot_div": plot_div,
            "table": forecast_future.to_dict(orient='records'),
            "download_path": f"/download/{filename}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail="Forecast failed: " + str(e))

@app.get("/download/{fname}")
async def download_file(fname: str):
    fpath = os.path.join(FORECAST_DIR, fname)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fpath, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=fname)
