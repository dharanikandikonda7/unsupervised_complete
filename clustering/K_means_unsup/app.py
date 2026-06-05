import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"
IMAGE_PATH = BASE_DIR / "images" / "cluster_plot.png"

# --------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Segmentation using K-Means")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

# --------------------------------------------------
# CLUSTER IMAGE
# --------------------------------------------------

if IMAGE_PATH.exists():
    st.image(IMAGE_PATH)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.subheader("Predict Customer Cluster")

income = st.number_input(
    "Income",
    value=50000
)

kidhome = st.number_input(
    "Kidhome",
    value=0
)

teenhome = st.number_input(
    "Teenhome",
    value=0
)

recency = st.number_input(
    "Recency",
    value=30
)

wines = st.number_input(
    "Wine Spending",
    value=100
)

fruits = st.number_input(
    "Fruit Spending",
    value=20
)

meat = st.number_input(
    "Meat Spending",
    value=100
)

fish = st.number_input(
    "Fish Spending",
    value=20
)

sweet = st.number_input(
    "Sweet Spending",
    value=20
)

gold = st.number_input(
    "Gold Spending",
    value=30
)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button("Predict Cluster"):

    sample = np.array([
        [
            income,
            kidhome,
            teenhome,
            recency,
            wines,
            fruits,
            meat,
            fish,
            sweet,
            gold
        ]
    ])

    sample_scaled = scaler.transform(sample)

    cluster = model.predict(sample_scaled)

    st.success(
        f"Customer belongs to Cluster {cluster[0]}"
    )