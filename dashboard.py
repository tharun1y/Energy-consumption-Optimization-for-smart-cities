import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO

st.title("Smart Grid Energy Optimization Dashboard")

# Load models
lstm_model = load_model("lstm_model.keras")
rl_model = PPO.load("ppo_smartgrid_model")

st.header("Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Detect numeric demand column
    numeric_cols = data.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        st.error("Dataset must contain a numeric demand column.")
        st.stop()

    demand_values = data[numeric_cols[0]].dropna().values

    if len(demand_values) < 7:
        st.error("Dataset must contain at least 7 numeric values.")
        st.stop()

    # LSTM prediction
    last_sequence = np.array(demand_values[-7:], dtype=float)
    last_sequence = last_sequence.reshape(1,7,1)

    predicted = lstm_model.predict(last_sequence)
    predicted_demand = float(predicted[0][0])

    st.header("Predicted Demand")
    st.metric("Predicted Demand (kWh)", round(predicted_demand,2))

    # RL state (same format used during training)
    state = np.array([0,0,0,predicted_demand], dtype=np.float32)

    action, _ = rl_model.predict(state)

    # Ensure valid distribution
    action = np.clip(action, 0, None)
    action = action / (np.sum(action) + 1e-8)

    # Energy allocation
    solar = predicted_demand * action[0]
    wind = predicted_demand * action[1]
    battery = predicted_demand * action[2]

    grid = max(predicted_demand - (solar + wind + battery), 0)

    st.header("Energy Distribution")

    st.write("Solar:", round(solar,2))
    st.write("Wind:", round(wind,2))
    st.write("Battery:", round(battery,2))
    st.write("Grid:", round(grid,2))

    # Pie chart
    labels = ["Solar","Wind","Battery","Grid"]
    values = [solar, wind, battery, grid]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title("Energy Source Distribution")

    st.pyplot(fig)

    # Performance metrics
    st.header("Optimization Performance")

    grid_cost = 8

    baseline_cost = predicted_demand * grid_cost
    optimized_cost = grid * grid_cost

    cost_reduction = ((baseline_cost - optimized_cost) / baseline_cost) * 100
    renewable_usage = ((solar + wind + battery) / predicted_demand) * 100

    col1, col2 = st.columns(2)

    col1.metric("Cost Reduction (%)", round(cost_reduction,2))
    col2.metric("Renewable Energy Usage (%)", round(renewable_usage,2))