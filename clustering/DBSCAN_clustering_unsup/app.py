import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="DBSCAN Clustering Dashboard",
    layout="wide"
)

st.title("📊 DBSCAN Customer Segmentation Dashboard")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/marketing_campaign.csv",
        sep="\t"
    )

df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "EDA",
        "Preprocessing",
        "DBSCAN Clustering",
        "PCA Visualization",
        "Noise Detection",
        "Cluster Analysis",
        "Insights"
    ]
)

# ---------------------------------------------------
# PREPROCESSING FUNCTION
# ---------------------------------------------------

@st.cache_data
def preprocess_data(data):

    data = data.copy()

    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    data["Age"] = 2025 - data["Year_Birth"]

    cat_cols = data.select_dtypes(
        include="object"
    ).columns

    le = LabelEncoder()

    for col in cat_cols:
        data[col] = le.fit_transform(
            data[col].astype(str)
        )

    data.fillna(
        data.median(numeric_only=True),
        inplace=True
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    return data, X

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write("Shape")

    st.write(df.shape)

    st.write("Columns")

    st.write(df.columns.tolist())

    st.write("Sample Data")

    st.dataframe(df.head())

    st.write("Missing Values")

    st.dataframe(
        df.isnull().sum()
    )

    st.write("Statistics")

    st.dataframe(
        df.describe()
    )

# ---------------------------------------------------
# EDA
# ---------------------------------------------------

elif page == "EDA":

    st.header("Exploratory Data Analysis")

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

    st.subheader(
        "Correlation Heatmap"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        df[numeric_cols].corr(),
        cmap="coolwarm"
    )

    st.pyplot(fig2)

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

elif page == "Preprocessing":

    st.header("Preprocessing")

    processed_data, X = preprocess_data(df)

    st.write(
        "Processed Dataset"
    )

    st.dataframe(
        processed_data.head()
    )

    st.write(
        "Final Shape:",
        processed_data.shape
    )

# ---------------------------------------------------
# DBSCAN CLUSTERING
# ---------------------------------------------------

elif page == "DBSCAN Clustering":

    st.header("DBSCAN Model")

    processed_data, X = preprocess_data(df)

    eps = st.slider(
        "EPS",
        0.1,
        5.0,
        1.5,
        0.1
    )

    min_samples = st.slider(
        "Min Samples",
        2,
        20,
        5
    )

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X)

    processed_data["Cluster"] = labels

    n_clusters = len(
        set(labels)
    ) - (
        1 if -1 in labels else 0
    )

    noise_count = list(labels).count(-1)

    col1, col2 = st.columns(2)

    col1.metric(
        "Clusters Found",
        n_clusters
    )

    col2.metric(
        "Noise Points",
        noise_count
    )

    if n_clusters > 1:
        score = silhouette_score(
            X,
            labels
        )

        st.success(
            f"Silhouette Score: {score:.4f}"
        )

    st.subheader(
        "Cluster Distribution"
    )

    st.dataframe(
        processed_data["Cluster"]
        .value_counts()
        .reset_index()
    )

# ---------------------------------------------------
# PCA VISUALIZATION
# ---------------------------------------------------

elif page == "PCA Visualization":

    st.header(
        "PCA Cluster Visualization"
    )

    processed_data, X = preprocess_data(df)

    eps = 1.5
    min_samples = 5

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X)

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Cluster": labels
        }
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10"
    )

    st.pyplot(fig)

# ---------------------------------------------------
# NOISE DETECTION
# ---------------------------------------------------

elif page == "Noise Detection":

    st.header(
        "Outlier Detection"
    )

    processed_data, X = preprocess_data(df)

    dbscan = DBSCAN(
        eps=1.5,
        min_samples=5
    )

    labels = dbscan.fit_predict(X)

    processed_data["Cluster"] = labels

    noise = processed_data[
        processed_data["Cluster"] == -1
    ]

    st.metric(
        "Noise Customers",
        len(noise)
    )

    st.dataframe(
        noise.head(20)
    )

# ---------------------------------------------------
# CLUSTER ANALYSIS
# ---------------------------------------------------

elif page == "Cluster Analysis":

    st.header(
        "Cluster Profile"
    )

    processed_data, X = preprocess_data(df)

    dbscan = DBSCAN(
        eps=1.5,
        min_samples=5
    )

    labels = dbscan.fit_predict(X)

    processed_data["Cluster"] = labels

    profile = (
        processed_data
        .groupby("Cluster")
        .mean()
        .round(2)
    )

    st.dataframe(profile)

# ---------------------------------------------------
# INSIGHTS
# ---------------------------------------------------

elif page == "Insights":

    st.header(
        "Business Insights"
    )

    st.markdown("""
### Key Findings

✅ DBSCAN automatically discovers customer groups.

✅ Customers labeled -1 are anomalies.

✅ Noise points may indicate:
- Unusual spending patterns
- Rare customer behavior
- Potential fraud cases

✅ Useful for:
- Customer segmentation
- Outlier detection
- Marketing targeting

✅ No need to specify number of clusters.
""")