import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

st.set_page_config(page_title="Supply Chain Coordination System", layout="wide")

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    lstm = load_model("lstm_demand_model.h5")
    risk_nn = load_model("delivery_risk_nn.h5")
    risk_scaler = joblib.load("risk_scaler.pkl")
    demand_scaler = joblib.load("demand_scaler.pkl")
    data = pd.read_csv("processed_supply_chain_data.csv")
    return lstm, risk_nn, risk_scaler, demand_scaler, data

lstm, risk_nn, risk_scaler, demand_scaler, df = load_artifacts()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Demand Forecast", "Bullwhip Analysis", "Delivery Risk", "Geographic Insights", "Recommendations"]
)

# =========================
# HOME
# =========================
if page == "Home":
    st.title("📦 Supply Chain Coordination System")
    st.write(
        """
        This system integrates **Demand Forecasting (LSTM)**, **Delivery Risk Prediction (NN)**,
        **Bullwhip Effect Analysis**, and **Geographic Insights** to support coordinated decisions.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Regions", df["Order Region"].nunique())
    c3.metric("Late Delivery Rate", f"{df['Late_delivery_risk'].mean()*100:.1f}%")

# =========================
# DEMAND FORECAST
# =========================
elif page == "Demand Forecast":
    st.title("📊 Demand Forecast (LSTM)")
    st.write("Forecast next demand using recent historical demand.")

    # Prepare time series
    ts = (
        df.groupby("Order Date")["Order Item Quantity"]
          .sum()
          .reset_index()
          .sort_values("Order Date")
    )

    lookback = st.slider("Lookback Window (days)", 7, 30, 14)

    scaled = demand_scaler.transform(ts[["Order Item Quantity"]].values)
    X = []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i+lookback])
    X = np.array(X).reshape(-1, lookback, 1)

    last_seq = X[-1:].copy()
    pred = lstm.predict(last_seq)
    pred_inv = demand_scaler.inverse_transform(pred)[0][0]

    st.metric("Predicted Next Demand", f"{pred_inv:.2f}")

    # Plot
    st.subheader("Recent Demand Trend")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(ts["Order Item Quantity"].tail(100).values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Quantity")
    st.pyplot(fig)

# =========================
# BULLWHIP ANALYSIS
# =========================
elif page == "Bullwhip Analysis":
    st.title("🌊 Bullwhip Effect Analysis")

    cust_var = df.groupby("Customer Segment")["Order Item Quantity"].var().mean()
    reg_var = df.groupby("Order Region")["Order Item Quantity"].var().mean()
    mar_var = df.groupby("Market")["Order Item Quantity"].var().mean()

    bw_reg = reg_var / cust_var
    bw_mar = mar_var / reg_var

    c1, c2 = st.columns(2)
    c1.metric("Region / Customer Variance Ratio", f"{bw_reg:.2f}")
    c2.metric("Market / Region Variance Ratio", f"{bw_mar:.2f}")

    st.subheader("Variance Amplification")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(["Customer", "Region", "Market"], [cust_var, reg_var, mar_var], marker="o")
    ax.set_ylabel("Variance")
    st.pyplot(fig)

# =========================
# DELIVERY RISK
# =========================
elif page == "Delivery Risk":
    st.title("🚚 Delivery Risk Predictor")

    c1, c2 = st.columns(2)
    qty = c1.number_input("Order Item Quantity", min_value=1, value=10)
    sales = c2.number_input("Sales per Customer", min_value=0.0, value=100.0)

    c3, c4 = st.columns(2)
    ship_real = c3.number_input("Days for Shipping (Real)", min_value=0, value=5)
    ship_sched = c4.number_input("Days for Shipment (Scheduled)", min_value=0, value=4)

    sample = np.array([[qty, sales, ship_real, ship_sched]])
    sample_scaled = risk_scaler.transform(sample)
    risk = float(risk_nn.predict(sample_scaled)[0][0])

    st.metric("Late Delivery Risk", f"{risk*100:.1f}%")

    if risk > 0.7:
        st.error("High Risk: Review shipping mode and buffer planning.")
    elif risk > 0.4:
        st.warning("Medium Risk: Monitor closely.")
    else:
        st.success("Low Risk: Operations stable.")

# =========================
# GEOGRAPHIC INSIGHTS
# =========================
elif page == "Geographic Insights":
    st.title("🗺️ Geographic Risk Heatmap")

    geo = df[["Latitude", "Longitude", "Late_delivery_risk"]].dropna()
    m = folium.Map(location=[geo["Latitude"].mean(), geo["Longitude"].mean()], zoom_start=2)
    HeatMap(geo[["Latitude", "Longitude", "Late_delivery_risk"]].values, radius=6).add_to(m)
    folium_static(m, width=900, height=500)

# =========================
# RECOMMENDATIONS
# =========================
elif page == "Recommendations":
    st.title("💡 Coordination Recommendations")

    avg_delay = (df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]).mean()
    late_rate = df["Late_delivery_risk"].mean()

    recs = []
    if late_rate > 0.3:
        recs.append("Increase coordination on shipping schedules.")
    if avg_delay > 0:
        recs.append("Reduce batch sizes to lower variability.")
    recs.append("Share real-time demand signals across tiers.")
    recs.append("Align replenishment cycles to demand forecasts.")

    for r in recs:
        st.write("•", r)
