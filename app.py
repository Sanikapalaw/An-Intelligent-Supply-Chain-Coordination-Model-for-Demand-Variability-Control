import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Coordination System",
    layout="wide"
)

st.title("📦 Intelligent Supply Chain Coordination System")
st.caption("Game-Theoretic Simulation of the Bullwhip Effect")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

mean_demand = st.sidebar.slider(
    "Average Customer Demand",
    min_value=100,
    max_value=1000,
    value=500,
    step=50
)

demand_variability = st.sidebar.slider(
    "Demand Variability",
    min_value=10,
    max_value=300,
    value=100,
    step=10
)

periods = st.sidebar.slider(
    "Number of Periods",
    min_value=30,
    max_value=120,
    value=60,
    step=10
)

scenario = st.sidebar.radio(
    "Coordination Strategy",
    ["Nash (Non-Cooperative)", "Pareto (Cooperative)"]
)

run_button = st.sidebar.button("▶ Run Simulation")

# --------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------
def generate_demand(mean, std, T):
    demand = np.random.normal(mean, std, T)
    return np.maximum(demand, 50)

def simulate_supply_chain(demand, mode):
    customer = demand

    if mode == "Nash (Non-Cooperative)":
        retailer = customer * 1.10
        distributor = retailer * 1.18
        manufacturer = distributor * 1.23
        supplier = manufacturer * 1.25
    else:
        retailer = customer * 1.02
        distributor = customer * 1.02
        manufacturer = customer * 1.02
        supplier = customer * 1.02

    return pd.DataFrame({
        "Customer": customer,
        "Retailer": retailer,
        "Distributor": distributor,
        "Manufacturer": manufacturer,
        "Supplier": supplier
    })

def variance(series):
    return np.var(series)

# --------------------------------------------------
# RUN SIMULATION
# --------------------------------------------------
if run_button:
    demand = generate_demand(mean_demand, demand_variability, periods)
    df = simulate_supply_chain(demand, scenario)

    # --------------------------------------------------
    # KPI METRICS
    # --------------------------------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    tiers = df.columns.tolist()

    for i, col in enumerate([col1, col2, col3, col4, col5]):
        amp = df[tiers[i]].iloc[-1] / df["Customer"].iloc[-1]
        col.metric(
            tiers[i],
            f"{int(df[tiers[i]].iloc[-1])}",
            f"{amp:.2f}×"
        )

    # --------------------------------------------------
    # ORDER QUANTITY OVER TIME
    # --------------------------------------------------
    st.subheader("📈 Order Quantities Across Supply Chain")

    fig, ax = plt.subplots(figsize=(10, 4))
    for tier in df.columns:
        ax.plot(df[tier], label=tier)

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Order Quantity")
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------
    # BULLWHIP EFFECT (VARIANCE)
    # --------------------------------------------------
    st.subheader("🌊 Bullwhip Effect (Variance Amplification)")

    variances = [variance(df[t]) for t in df.columns]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(df.columns, variances, color="orange")
    ax2.set_ylabel("Variance")
    ax2.set_title("Variance Increases Upstream")
    st.pyplot(fig2)

    # --------------------------------------------------
    # ANALYSIS SECTION
    # --------------------------------------------------
    st.subheader("📊 Analysis")

    if scenario == "Nash (Non-Cooperative)":
        st.error(
            "Non-cooperative behavior leads to high demand amplification "
            "as each tier adds safety stock independently."
        )
    else:
        st.success(
            "Coordinated decision-making significantly reduces variability "
            "and stabilizes the supply chain."
        )

    # --------------------------------------------------
    # GAME THEORY EXPLANATION
    # --------------------------------------------------
    st.subheader("🎮 Game Theory Interpretation")

    st.markdown("""
    **Nash Equilibrium (Non-Cooperative):**
    - Each tier optimizes locally
    - Order inflation becomes a dominant strategy
    - Results in the bullwhip effect

    **Pareto Optimal (Cooperative):**
    - Information sharing across tiers
    - Lower safety stock buffers
    - Higher total system efficiency
    """)

else:
    st.info("Adjust parameters and click **Run Simulation** to start.")
