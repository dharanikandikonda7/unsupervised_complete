import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering",
    layout="wide"
)

st.title("📊 Customer Segmentation using Hierarchical Clustering")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
    return df

df = load_data()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "EDA",
        "Preprocessing",
        "Dendrogram",
        "Clustering",
        "Insights"
    ]
)

# ====================================================
# DATASET OVERVIEW
# ====================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Data Types")

    st.dataframe(df.dtypes.astype(str))

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# ====================================================
# EDA
# ====================================================

elif page == "EDA":

    st.header("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns

    selected_col = st.selectbox(
        "Select Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[selected_col],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(12,8))

    sns.heatmap(
        df[numeric_cols].corr(),
        cmap="coolwarm"
    )

    st.pyplot(fig2)

# ====================================================
# PREPROCESSING
# ====================================================

elif page == "Preprocessing":

    st.header("Data Preprocessing")

    df2 = df.copy()

    st.subheader("Original Shape")

    st.write(df2.shape)

    if "ID" in df2.columns:
        df2.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in df2.columns:
        df2.drop("Dt_Customer", axis=1, inplace=True)

    categorical = df2.select_dtypes(include="object").columns

    le = LabelEncoder()

    for col in categorical:
        df2[col] = le.fit_transform(df2[col].astype(str))

    df2.fillna(df2.median(numeric_only=True), inplace=True)

    st.subheader("Processed Data")

    st.dataframe(df2.head())

    st.write("Shape:", df2.shape)

# ====================================================
# DENDROGRAM
# ====================================================

elif page == "Dendrogram":

    st.header("Hierarchical Dendrogram")

    data = df.copy()

    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    cat_cols = data.select_dtypes(include="object").columns

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

    sample_size = st.slider(
        "Select Samples for Dendrogram",
        50,
        500,
        200
    )

    X_sample = X[:sample_size]

    linkage_matrix = linkage(
        X_sample,
        method='ward'
    )

    fig, ax = plt.subplots(
        figsize=(15,7)
    )

    dendrogram(linkage_matrix)

    plt.title("Dendrogram")

    st.pyplot(fig)

# ====================================================
# CLUSTERING
# ====================================================

elif page == "Clustering":

    st.header("Agglomerative Clustering")

    data = df.copy()

    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    cat_cols = data.select_dtypes(include="object").columns

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

    n_clusters = st.slider(
        "Number of Clusters",
        2,
        10,
        3
    )

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels
    )

    st.success(
        f"Silhouette Score : {score:.4f}"
    )

    data["Cluster"] = labels

    st.subheader("Cluster Counts")

    st.dataframe(
        data["Cluster"]
        .value_counts()
        .reset_index()
    )

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": pca_result[:,0],
            "PC2": pca_result[:,1],
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
        palette="Set1"
    )

    st.pyplot(fig)

# ====================================================
# INSIGHTS
# ====================================================

elif page == "Insights":

    st.header("Cluster Insights")

    data = df.copy()

    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    cat_cols = data.select_dtypes(include="object").columns

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

    model = AgglomerativeClustering(
        n_clusters=3
    )

    labels = model.fit_predict(X)

    data["Cluster"] = labels

    st.subheader(
        "Cluster Mean Comparison"
    )

    cluster_profile = (
        data.groupby("Cluster")
        .mean()
        .round(2)
    )

    st.dataframe(cluster_profile)

    st.subheader(
        "Average Income per Cluster"
    )

    if "Income" in data.columns:

        fig, ax = plt.subplots()

        cluster_profile["Income"].plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)