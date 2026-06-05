import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import joblib
import streamlit as st 


st.set_page_config(
    page_title="Customer Segmentation ",
    layout="wide"
)

st.title(
    "Customer Segmentation using k-means"
)

model=joblib.load(
    "models/kmeans_model.pkl"
)

scaler=joblib.load(
    "models/scaler.pkl"
)

df = pd.read_csv(
    "data/marketing_campaign.csv",
    sep="\t"
)
st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head()
)

st.image(
    "images/cluster_plot.png"
)

st.subheader(
    "Predict Customer Cluster"
)

income=st.number_input(
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
    value=30)

if st.button(
    "Predict Cluster"
):
    sample=np.array([[
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
    ]])
    sample_scaled=scaler.transform(
        sample
    )
    cluster=model.predict(
        sample_scaled
    )
    st.success(
        f"Customer belongs to Cluster {cluster[0]}"
    )