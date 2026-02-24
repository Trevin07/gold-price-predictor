#  🪙Gold Price Forecasting System

## Executive Summary
This project implements a high-precision time-series forecasting engine for short-term financial trend analysis.  
Using Facebook Prophet, the system generates probabilistic forecasts with uncertainty intervals to support financial planning and collateral risk assessment.

---

## Key Capabilities

### Time-Series Modeling
Built an additive forecasting model capturing non-linear trends with:
- Yearly seasonality
- Weekly seasonality
- Daily seasonality

### Risk Mitigation
Provides prediction intervals that quantify market volatility, enabling scenario-based decision making.

### Automated Insights
End-to-end pipeline that:
1. Ingests historical price data
2. Cleans and preprocesses it
3. Produces interpretable trend forecasts

---

## Technical Stack

**Modeling**
- Facebook Prophet

**Data Engineering**
- Python
- Pandas
- NumPy

**Analytics**
- Statistical Inference
- Regression Analysis

**Visualization**
- Plotly
- Matplotlib

---

## Project Structure

```
├── data/               # Historical gold price datasets
├── notebooks/          # EDA & Hyperparameter tuning
├── src/
│   ├── preprocess.py   # Data cleaning and feature engineering
│   └── forecast.py     # Model training and prediction
├── requirements.txt    # Dependencies
└── README.md
```

---

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Trevin07/gold-price-predictor.git
cd gold-price-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Forecast
```bash
python src/forecast.py
```

---

## Output
The system generates:
- Forecasted gold price trend
- Confidence intervals
- Trend decomposition plots

---

## Use Cases
- Financial planning
- Collateral valuation
- Market trend monitoring
- Risk analysis
