# =============================================================================
# Predictive Liquidity & AR Manager
# Author: Jonas Nikolaus Felder
# Last updated: 2026
# Python: B2B Data Simulation, ML Forecast + Customer Risk Scoring
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# Adjust the path to your system if necessary
DB_PATH = r'C:\Users\jonas\Documents\Projects\Predictive Liquidity & AR (Accounts Receivable) Manager\Liquidity_AR.db'
OUT_DIR = "."

print("=" * 65)
print("  Liquidity & AR Manager  |  Phase 2: ML Forecast (B2B Version)")
print("=" * 65)

# -----------------------------------------------------------------------------
# 1. LOAD DATA & SIMULATE B2B PAYMENT BEHAVIOUR
# -----------------------------------------------------------------------------
print("\n[1/4] Loading data & simulating B2B payment behaviour...")

conn = sqlite3.connect(DB_PATH)

# We only load the base data; the date logic is rebuilt in Python
df = pd.read_sql("""
    SELECT 
        order_id, 
        customer_unique_id, 
        region, 
        order_date, 
        invoice_amount, 
        payment_type, 
        payment_installments
    FROM ar_reporting_base 
    WHERE invoice_amount > 0
""", conn)

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# --- THE B2B SIMULATION ---
# 1. Standard payment terms: 30 days after order
df["payment_due_date"] = df["order_date"] + pd.Timedelta(days=30)

# 2. Simulate realistic delay
#    60% pay on time (up to 3 days early)
#    25% pay slightly late (1 to 15 days)
#    15% pay critically late (16 to 60 days)
def simulate_b2b_delay(n):
    rands = np.random.rand(n)
    delays = np.zeros(n)
    
    mask_on_time = rands < 0.60
    delays[mask_on_time] = np.random.randint(-3, 1, size=mask_on_time.sum())
    
    mask_late = (rands >= 0.60) & (rands < 0.85)
    delays[mask_late] = np.random.randint(1, 16, size=mask_late.sum())
    
    mask_critical = rands >= 0.85
    delays[mask_critical] = np.random.randint(16, 61, size=mask_critical.sum())
    return delays

df["payment_delay"] = simulate_b2b_delay(len(df))

# ML Feature Engineering: We force correlations into the data so the ML model has something to learn
# High installments or specific regions artificially increase the delay
df.loc[df["payment_installments"] > 5, "payment_delay"] += np.random.randint(5, 20, size=(df["payment_installments"] > 5).sum())
df.loc[df["region"].isin(["RJ", "CE", "MA"]), "payment_delay"] += np.random.randint(10, 30, size=df["region"].isin(["RJ", "CE", "MA"]).sum())

# 3. Calculate actual payment date and flags
df["payment_received_date"] = df["payment_due_date"] + pd.to_timedelta(df["payment_delay"], unit="D")
df["is_late"] = (df["payment_delay"] > 0).astype(int)

# 4. Ageing Buckets for Power BI
bins = [-np.inf, 0, 30, 60, 90, np.inf]
labels = ["On Time", "1-30 Days", "31-60 Days", "61-90 Days", "90+ Days"]
df["ageing_bucket"] = pd.cut(df["payment_delay"], bins=bins, labels=labels)

# Save in SQLite as a new table for Power BI
df.to_sql("ar_reporting_base_b2b", conn, if_exists="replace", index=False)
df.to_csv("ar_reporting_base_b2b.csv", index=False)
conn.close()

print(f"   {len(df):,} rows processed")
print(f"   Average B2B payment delay: {df['payment_delay'].mean():.1f} days")
print(f"   Overall Late Payment Rate: {df['is_late'].mean()*100:.1f}%")


# -----------------------------------------------------------------------------
# 2. CASHFLOW TIME SERIES (Using simulated payment dates)
# -----------------------------------------------------------------------------
print("\n[2/4] Building daily cashflow time series...")

cash_ts = (
    df.dropna(subset=["payment_received_date"])
    .groupby(df["payment_received_date"].dt.date)["invoice_amount"]
    .sum()
    .reset_index()
    .rename(columns={"payment_received_date": "ds", "invoice_amount": "y"})
    .sort_values("ds")
)

cash_ts["ds"] = pd.to_datetime(cash_ts["ds"])
full_range = pd.date_range(cash_ts["ds"].min(), cash_ts["ds"].max(), freq="D")
cash_ts = (
    cash_ts.set_index("ds")
    .reindex(full_range, fill_value=0.0)
    .rename_axis("ds")
    .reset_index()
)

# 7-day rolling average
cash_ts["y_smooth"] = cash_ts["y"].rolling(7, center=True, min_periods=1).mean()

print(f"   {len(cash_ts)} days of history")
print(f"   Avg daily cash-in:   R${cash_ts['y'].mean():>10,.0f}")


# -----------------------------------------------------------------------------
# 3. HOLT-WINTERS TRIPLE EXPONENTIAL SMOOTHING  (90-day forecast)
# -----------------------------------------------------------------------------
print("\n[3/4] Running Holt-Winters forecast (90-day horizon)...")

def holt_winters_forecast(series, alpha=0.25, beta=0.08, gamma=0.15, season_len=7, n_forecast=90):
    y   = np.array(series, dtype=float)
    n   = len(y)
    eps = 1e-9

    level  = float(np.mean(y[:season_len]))
    trend  = float((np.mean(y[season_len:2*season_len]) - np.mean(y[:season_len])) / season_len)
    season = list(y[:season_len] / (level + eps))

    levels, trends, fitted = [level], [trend], []

    for t in range(n):
        s_idx  = t % season_len
        s_prev = season[s_idx]
        l_prev = levels[-1]
        b_prev = trends[-1]

        l_new  = alpha * (y[t] / (s_prev + eps)) + (1 - alpha) * (l_prev + b_prev)
        b_new  = beta  * (l_new - l_prev)        + (1 - beta)  * b_prev
        s_new  = gamma * (y[t] / (l_new + eps))   + (1 - gamma) * s_prev

        levels.append(l_new)
        trends.append(b_new)
        season[s_idx] = s_new
        fitted.append(max(0.0, (l_new + b_new) * s_new))

    forecast = []
    for h in range(1, n_forecast + 1):
        l_h = levels[-1] + h * trends[-1]
        s_h = season[(n + h - 1) % season_len]
        forecast.append(max(0.0, l_h * s_h))

    return np.array(fitted), np.array(forecast)

fitted_vals, forecast_vals = holt_winters_forecast(cash_ts["y_smooth"].values)
actual_fit = cash_ts["y_smooth"].values[:len(fitted_vals)]

last_date      = cash_ts["ds"].max()
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90, freq="D")
residuals      = cash_ts["y_smooth"].values[-60:] - fitted_vals[-60:]
residual_std   = float(np.std(residuals))

cashflow_forecast = pd.DataFrame({
    "Date":              forecast_dates.strftime("%Y-%m-%d"),
    "Predicted_Cash_In": np.round(forecast_vals, 2),
    "Lower_80":          np.round(np.maximum(0, forecast_vals - 1.28 * residual_std), 2),
    "Upper_80":          np.round(forecast_vals + 1.28 * residual_std, 2),
})

forecast_path = os.path.join(OUT_DIR, "cashflow_forecast.csv")
cashflow_forecast.to_csv(forecast_path, index=False)
print(f"   90-day forecast exported  ->  {forecast_path}")


# -----------------------------------------------------------------------------
# 4. CUSTOMER RISK SCORING (Gradient Boosting)
# -----------------------------------------------------------------------------
print("\n[4/4] Building ML customer risk scores...")

cust = (
    df.groupby("customer_unique_id")
    .agg(
        order_count         = ("order_id",             "nunique"),
        total_spend         = ("invoice_amount",       "sum"),
        avg_payment_delay   = ("payment_delay",        "mean"), # Our Target
        installment_ratio   = ("payment_installments", lambda x: (x > 1).mean()),
        avg_installments    = ("payment_installments", "mean"),
        region              = ("region",               "first"),
    )
    .reset_index()
)

cust["log_spend"] = np.log1p(cust["total_spend"])
top_regions = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO"]
cust["region_encoded"] = cust["region"].apply(
    lambda r: top_regions.index(r) if r in top_regions else len(top_regions)
)

# INDEPENDENT features (no target leakage!)
FEATURES = ["order_count", "log_spend", "installment_ratio", "avg_installments", "region_encoded"]

X     = cust[FEATURES].fillna(0).values
y_reg = cust["avg_payment_delay"].fillna(0).values

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

gbm = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
gbm.fit(X_train, y_train)

# Predict expected delay for ALL customers to build the score
predicted_delay = gbm.predict(X)
scaler          = MinMaxScaler(feature_range=(0, 100))
risk_scores     = scaler.fit_transform(predicted_delay.reshape(-1, 1)).flatten()

cust["risk_score"] = np.round(risk_scores, 1)
cust["risk_tier"]  = pd.cut(
    cust["risk_score"],
    bins   = [0, 25, 50, 75, 100],
    labels = ["Low", "Medium", "High", "Critical"],
    include_lowest=True
)

risk_output = cust[[
    "customer_unique_id", "region", "order_count", "total_spend", 
    "avg_installments", "risk_score", "risk_tier"
]].copy()

risk_path = os.path.join(OUT_DIR, "customer_risk_scores.csv")
risk_output.to_csv(risk_path, index=False)

print(f"   {len(risk_output):,} customers scored.")
print(f"   Risk scores exported -> {risk_path}")

print("\n" + "=" * 65)
print("  Done. Update your Power BI data source to point to:")
print("  - ar_reporting_base_b2b.csv (or the updated SQLite table)")
print("  - cashflow_forecast.csv")
print("  - customer_risk_scores.csv")
print("=" * 65 + "\n")