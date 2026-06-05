import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        sep="\t"
    )

st.title("📊 Principal Component Analysis (PCA) Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"

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

    # Remove unwanted columns
    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    # Feature Engineering
    data["Age"] = 2025 - data["Year_Birth"]

    data["Total_Spending"] = (
        data["MntWines"]
        + data["MntFruits"]
        + data["MntMeatProducts"]
        + data["MntFishProducts"]
        + data["MntSweetProducts"]
        + data["MntGoldProds"]
    )

    # Encode categorical columns
    cat_cols = data.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(
            data[col].astype(str)
        )

    # Missing values
    data.fillna(
        data.median(numeric_only=True),
        inplace=True
    )

    # Scaling
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
        "Explained Variance",
        "Cumulative Variance",
        "Feature Contributions",
        "2D PCA Projection",
        "3D PCA Projection",
        "Insights"
    ]
)

# ==================================================
# DATASET OVERVIEW
# ==================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write("Shape:", df.shape)

    st.subheader("Columns")
    st.write(df.columns.tolist())

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Statistics")
    st.dataframe(df.describe())

# ==================================================
# EDA
# ==================================================

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

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        df[numeric_cols].corr(),
        cmap="coolwarm"
    )

    st.pyplot(fig2)

# ==================================================
# PREPROCESSING
# ==================================================

elif page == "Preprocessing":

    st.header("Processed Dataset")

    processed_data, X = preprocess_data(df)

    st.write(
        "Shape:",
        processed_data.shape
    )

    st.dataframe(
        processed_data.head()
    )

# ==================================================
# EXPLAINED VARIANCE
# ==================================================

elif page == "Explained Variance":

    st.header("Explained Variance Analysis")

    processed_data, X = preprocess_data(df)

    pca = PCA()
    pca.fit(X)

    variance_ratio = (
        pca.explained_variance_ratio_
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.bar(
        range(
            1,
            len(variance_ratio)+1
        ),
        variance_ratio
    )

    ax.set_xlabel(
        "Principal Components"
    )

    ax.set_ylabel(
        "Explained Variance"
    )

    st.pyplot(fig)

# ==================================================
# CUMULATIVE VARIANCE
# ==================================================

elif page == "Cumulative Variance":

    st.header("Cumulative Variance")

    processed_data, X = preprocess_data(df)

    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        cumulative_variance,
        marker='o'
    )

    ax.axhline(
        y=0.90,
        color='red',
        linestyle='--'
    )

    ax.set_xlabel(
        "Components"
    )

    ax.set_ylabel(
        "Cumulative Variance"
    )

    st.pyplot(fig)

    components_90 = (
        np.argmax(
            cumulative_variance >= 0.90
        ) + 1
    )

    st.success(
        f"{components_90} components retain 90% variance."
    )

# ==================================================
# FEATURE CONTRIBUTIONS
# ==================================================

elif page == "Feature Contributions":

    st.header("Feature Contributions")

    processed_data, X = preprocess_data(df)

    pca = PCA()
    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[
            f"PC{i+1}"
            for i in range(
                pca.n_components_
            )
        ],
        index=processed_data.columns
    )

    st.dataframe(
        loadings.iloc[:, :5]
    )

    st.subheader(
        "Heatmap (First 5 PCs)"
    )

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        loadings.iloc[:, :5],
        cmap="coolwarm",
        center=0
    )

    st.pyplot(fig)

# ==================================================
# 2D PCA
# ==================================================

elif page == "2D PCA Projection":

    st.header("2D PCA Projection")

    processed_data, X = preprocess_data(df)

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1]
        }
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2"
    )

    st.pyplot(fig)

# ==================================================
# 3D PCA
# ==================================================

elif page == "3D PCA Projection":

    st.header("3D PCA Projection")

    processed_data, X = preprocess_data(df)

    pca = PCA(
        n_components=3
    )

    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "PC3": X_pca[:,2]
        }
    )

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# INSIGHTS
# ==================================================

elif page == "Insights":

    st.header("Business Insights")

    processed_data, X = preprocess_data(df)

    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    components_90 = (
        np.argmax(
            cumulative_variance >= 0.90
        ) + 1
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Features",
        X.shape[1]
    )

    col2.metric(
        "Components for 90%",
        components_90
    )

    col3.metric(
        "Variance Retained",
        "90%"
    )

    st.markdown("""
### Key Findings

✅ PCA reduces dimensionality.

✅ Removes redundant information.

✅ Preserves maximum variance.

✅ Helps visualize high-dimensional customer data.

### Interpretation

- PC1 captures maximum variance.
- PC2 captures second highest variance.
- Customer behavior can be analyzed using fewer dimensions.
- PCA can improve clustering performance.

### Benefits

- Faster model training
- Noise reduction
- Better visualization
- Reduced complexity
""")