import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="GMM Customer Segmentation",
    layout="wide"
)

st.title("📊 Gaussian Mixture Model (GMM) Customer Segmentation")

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

    # Drop unnecessary columns
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

    # Encoding
    cat_cols = data.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(
            data[col].astype(str)
        )

    # Missing Values
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
        "AIC-BIC Analysis",
        "GMM Clustering",
        "Probability Analysis",
        "PCA Visualization",
        "Cluster Profiling",
        "Insights"
    ]
)

# ==================================================
# DATASET OVERVIEW
# ==================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write("Shape:", df.shape)

    st.write("Columns")
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

    selected_feature = st.selectbox(
        "Select Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[selected_feature],
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

    st.header("Preprocessed Data")

    processed_data, X = preprocess_data(df)

    st.write(
        "Shape:",
        processed_data.shape
    )

    st.dataframe(
        processed_data.head()
    )

# ==================================================
# AIC / BIC
# ==================================================

elif page == "AIC-BIC Analysis":

    st.header("Optimal Components Selection")

    processed_data, X = preprocess_data(df)

    aic = []
    bic = []

    components = range(2, 11)

    for k in components:

        model = GaussianMixture(
            n_components=k,
            random_state=42
        )

        model.fit(X)

        aic.append(model.aic(X))
        bic.append(model.bic(X))

    fig, ax = plt.subplots()

    ax.plot(
        components,
        aic,
        marker='o',
        label='AIC'
    )

    ax.plot(
        components,
        bic,
        marker='o',
        label='BIC'
    )

    ax.legend()

    ax.set_xlabel("Components")
    ax.set_ylabel("Score")

    st.pyplot(fig)

    best_k = components[
        np.argmin(bic)
    ]

    st.success(
        f"Best Components based on BIC: {best_k}"
    )

# ==================================================
# GMM CLUSTERING
# ==================================================

elif page == "GMM Clustering":

    st.header("Gaussian Mixture Clustering")

    processed_data, X = preprocess_data(df)

    n_components = st.slider(
        "Number of Components",
        2,
        10,
        3
    )

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42
    )

    labels = gmm.fit_predict(X)

    processed_data["Cluster"] = labels

    score = silhouette_score(
        X,
        labels
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Clusters",
        n_components
    )

    col2.metric(
        "AIC",
        round(gmm.aic(X))
    )

    col3.metric(
        "BIC",
        round(gmm.bic(X))
    )

    st.success(
        f"Silhouette Score : {score:.4f}"
    )

    st.subheader(
        "Cluster Distribution"
    )

    st.dataframe(
        processed_data["Cluster"]
        .value_counts()
        .reset_index()
    )

# ==================================================
# PROBABILITY ANALYSIS
# ==================================================

elif page == "Probability Analysis":

    st.header("Cluster Membership Probability")

    processed_data, X = preprocess_data(df)

    gmm = GaussianMixture(
        n_components=3,
        random_state=42
    )

    gmm.fit(X)

    probabilities = gmm.predict_proba(X)

    prob_df = pd.DataFrame(
        probabilities,
        columns=[
            f"Cluster_{i}"
            for i in range(
                probabilities.shape[1]
            )
        ]
    )

    st.dataframe(
        prob_df.head(20)
    )

    max_prob = np.max(
        probabilities,
        axis=1
    )

    fig, ax = plt.subplots()

    sns.histplot(
        max_prob,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Maximum Cluster Probability"
    )

    st.pyplot(fig)

# ==================================================
# PCA VISUALIZATION
# ==================================================

elif page == "PCA Visualization":

    st.header("PCA Cluster Visualization")

    processed_data, X = preprocess_data(df)

    gmm = GaussianMixture(
        n_components=3,
        random_state=42
    )

    labels = gmm.fit_predict(X)

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

# ==================================================
# CLUSTER PROFILING
# ==================================================

elif page == "Cluster Profiling":

    st.header("Cluster Profile Analysis")

    processed_data, X = preprocess_data(df)

    gmm = GaussianMixture(
        n_components=3,
        random_state=42
    )

    labels = gmm.fit_predict(X)

    processed_data["Cluster"] = labels

    profile = (
        processed_data
        .groupby("Cluster")
        .mean()
        .round(2)
    )

    st.dataframe(profile)

    st.subheader("Profile Heatmap")

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        profile,
        cmap="coolwarm"
    )

    st.pyplot(fig)

# ==================================================
# INSIGHTS
# ==================================================

elif page == "Insights":

    st.header("Business Insights")

    st.markdown("""
### Findings

✅ GMM groups customers based on probability distributions.

✅ Customers can partially belong to multiple clusters.

✅ Useful for customer segmentation and marketing.

### Example Segments

**Cluster 0**
- High Income
- High Spending

**Cluster 1**
- Medium Income
- Moderate Purchases

**Cluster 2**
- Low Income
- Less Active Customers

### Advantages of GMM

- Soft Clustering
- Handles Overlapping Clusters
- Better than K-Means for complex data

### Evaluation Metrics

- Silhouette Score
- AIC
- BIC
""")