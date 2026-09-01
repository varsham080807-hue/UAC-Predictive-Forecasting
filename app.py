
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("Decision-support dashboard for forecasting HHS care load and discharge demand")

@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    numeric = [c for c in df.columns if c != "Date"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("Date").asfreq("D")
    df[numeric] = df[numeric].interpolate(limit_direction="both")
    return df

df = load_data()
care_col = "Children in HHS Care"
discharge_col = "Children discharged from HHS Care"
transfer_col = "Children transferred out of CBP custody"

st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
model_name = st.sidebar.selectbox(
    "Model",
    ["Naive", "Moving Average", "Exponential Smoothing", "Random Forest", "Gradient Boosting"]
)

st.subheader("Current Program Snapshot")
latest = df.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("Children currently in HHS Care", f"{latest[care_col]:,.0f}")
c2.metric("Latest daily discharges", f"{latest[discharge_col]:,.0f}")
c3.metric("Latest net pressure", f"{latest[transfer_col]-latest[discharge_col]:,.0f}")

def ml_features(series):
    x = pd.DataFrame(index=series.index)
    x["lag1"] = series.shift(1)
    x["lag7"] = series.shift(7)
    x["lag14"] = series.shift(14)
    x["roll7"] = series.shift(1).rolling(7).mean()
    x["roll14"] = series.shift(1).rolling(14).mean()
    x["dow"] = series.index.dayofweek
    x["month"] = series.index.month
    x["y"] = series
    return x.dropna()

def forecast_ml(series, horizon, model):
    feat = ml_features(series)
    X = feat.drop(columns="y")
    y = feat["y"]
    model.fit(X, y)
    history = series.copy()
    preds = []
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    for d in dates:
        row = pd.DataFrame({
            "lag1":[history.iloc[-1]],
            "lag7":[history.iloc[-7] if len(history)>=7 else history.iloc[-1]],
            "lag14":[history.iloc[-14] if len(history)>=14 else history.iloc[-1]],
            "roll7":[history.tail(7).mean()],
            "roll14":[history.tail(14).mean()],
            "dow":[d.dayofweek],
            "month":[d.month]
        })
        p = max(0, float(model.predict(row)[0]))
        preds.append(p)
        history.loc[d] = p
    return pd.Series(preds, index=dates)

def make_forecast(series, horizon, name):
    series = series.dropna()
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if name == "Naive":
        pred = np.repeat(series.iloc[-1], horizon)
        return pd.Series(pred, index=dates)
    if name == "Moving Average":
        pred = np.repeat(series.tail(7).mean(), horizon)
        return pd.Series(pred, index=dates)
    if name == "Exponential Smoothing":
        fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        return fit.forecast(horizon)
    if name == "Random Forest":
        return forecast_ml(series, horizon, RandomForestRegressor(n_estimators=200, random_state=42))
    return forecast_ml(series, horizon, GradientBoostingRegressor(random_state=42))

care_forecast = make_forecast(df[care_col], horizon, model_name)
discharge_forecast = make_forecast(df[discharge_col], horizon, model_name)

st.subheader("Future Care Load Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-90:], y=df[care_col].tail(90), name="Historical"))
fig.add_trace(go.Scatter(x=care_forecast.index, y=care_forecast.values, name="Forecast"))
# simple uncertainty band based on recent volatility
sigma = df[care_col].diff().tail(30).std()
upper = care_forecast + 1.96*sigma*np.sqrt(np.arange(1, horizon+1))
lower = np.maximum(0, care_forecast - 1.96*sigma*np.sqrt(np.arange(1, horizon+1)))
fig.add_trace(go.Scatter(x=upper.index, y=upper, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=lower.index, y=lower, fill="tonexty", line=dict(width=0), name="Approx. 95% interval"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Discharge Demand Forecast")
st.line_chart(pd.DataFrame({"Historical": df[discharge_col].tail(60), "Forecast": discharge_forecast}))

net_pressure = care_forecast.diff().fillna(0)
risk = "LOW"
if net_pressure.max() > df[care_col].diff().tail(90).quantile(.9):
    risk = "HIGH"
elif net_pressure.max() > df[care_col].diff().tail(90).quantile(.7):
    risk = "MODERATE"

st.subheader("Early Warning Indicator")
st.warning(f"Capacity stress risk: **{risk}**")

st.subheader("Forecast Table")
out = pd.DataFrame({
    "Forecast Date": care_forecast.index.date,
    "Predicted HHS Care Load": np.round(care_forecast.values),
    "Predicted Discharges": np.round(discharge_forecast.values)
})
st.dataframe(out, use_container_width=True)
st.download_button("Download Forecast CSV", out.to_csv(index=False), "uac_forecast.csv", "text/csv")
