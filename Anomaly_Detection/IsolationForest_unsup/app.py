import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Isolation Forest Dashboard",
    layout="wide"
)

st.title(
    "🚨 Isolation Forest Anomaly Detection"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "isolation_forest.pkl"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "scaler.pkl"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH,
        sep="\t"
    )

df = load_data()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

@st.cache_data
def preprocess_data(df):

    data = df.copy()

    if "ID" in data.columns:
        data.drop(
            "ID",
            axis=1,
            inplace=True
        )

    if "Dt_Customer" in data.columns:
        data.drop(
            "Dt_Customer",
            axis=1,
            inplace=True
        )

    data["Age"] = (
        2025 - data["Year_Birth"]
    )

    data["Total_Spending"] = (
        data["MntWines"]
        + data["MntFruits"]
        + data["MntMeatProducts"]
        + data["MntFishProducts"]
        + data["MntSweetProducts"]
        + data["MntGoldProds"]
    )

    cat_cols = data.select_dtypes(
        include="object"
    ).columns

    for col in cat_cols:

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

    data.fillna(
        data.median(
            numeric_only=True
        ),
        inplace=True
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    return data, X

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "EDA",
        "Preprocessing",
        "Anomaly Detection",
        "PCA Visualization",
        "Anomaly Analysis",
        "Manual Detection",
        "Insights"
    ]
)

# ==================================================
# DATASET OVERVIEW
# ==================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write(df.shape)

    st.dataframe(
        df.head()
    )

    st.dataframe(
        df.describe()
    )

# ==================================================
# EDA
# ==================================================

elif page == "EDA":

    st.header("EDA")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    feature = st.selectbox(
        "Select Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[feature],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# ==================================================
# PREPROCESSING
# ==================================================

elif page == "Preprocessing":

    processed_data, X = preprocess_data(df)

    st.dataframe(
        processed_data.head()
    )

    st.write(
        processed_data.shape
    )

# ==================================================
# ANOMALY DETECTION
# ==================================================

elif page == "Anomaly Detection":

    st.header(
        "Isolation Forest"
    )

    processed_data, X = preprocess_data(df)

    contamination = st.slider(
        "Contamination",
        0.01,
        0.20,
        0.05
    )

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    labels = model.fit_predict(X)

    processed_data["Anomaly"] = labels

    anomalies = (
        labels == -1
    ).sum()

    normal = (
        labels == 1
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "Normal",
        normal
    )

    col2.metric(
        "Anomalies",
        anomalies
    )

    st.dataframe(
        processed_data[
            processed_data[
                "Anomaly"
            ] == -1
        ].head(20)
    )

# ==================================================
# PCA VISUALIZATION
# ==================================================

elif page == "PCA Visualization":

    processed_data, X = preprocess_data(df)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    labels = model.fit_predict(X)

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Anomaly": labels
        }
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Anomaly"
    )

    st.pyplot(fig)

# ==================================================
# ANOMALY ANALYSIS
# ==================================================

elif page == "Anomaly Analysis":

    processed_data, X = preprocess_data(df)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    labels = model.fit_predict(X)

    processed_data["Anomaly"] = labels

    anomaly_df = processed_data[
        processed_data["Anomaly"] == -1
    ]

    st.write(
        anomaly_df.describe()
    )

# ==================================================
# MANUAL DETECTION
# ==================================================

elif page == "Manual Detection":

    st.header(
        "Check Customer"
    )

    income = st.number_input(
        "Income",
        50000
    )

    spending = st.number_input(
        "Total Spending",
        500
    )

    if st.button(
        "Check"
    ):

        if income > 150000 or spending > 2000:

            st.error(
                "Potential Anomaly"
            )

        else:

            st.success(
                "Looks Normal"
            )

# ==================================================
# INSIGHTS
# ==================================================

elif page == "Insights":

    st.markdown("""
### Isolation Forest

✅ Detects unusual customers

✅ Works without labels

✅ Efficient for large datasets

### Business Uses

- Fraud Detection
- Customer Behavior Analysis
- Rare Event Detection
- Risk Monitoring

### Output Labels

1 → Normal

-1 → Anomaly
""")