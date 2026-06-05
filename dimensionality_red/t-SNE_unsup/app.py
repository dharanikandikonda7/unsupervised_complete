import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="t-SNE Dimensionality Reduction",
    layout="wide"
)

st.title("📊 t-SNE Dimensionality Reduction Dashboard")

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"

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
        data.drop("ID", axis=1, inplace=True)

    if "Dt_Customer" in data.columns:
        data.drop("Dt_Customer", axis=1, inplace=True)

    data["Age"] = 2025 - data["Year_Birth"]

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
        data.median(numeric_only=True),
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
        "PCA Comparison",
        "t-SNE Visualization",
        "Parameter Analysis",
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

    st.dataframe(
        df.isnull().sum()
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe()
    )

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
# PCA COMPARISON
# ==================================================

elif page == "PCA Comparison":

    st.header("PCA Baseline")

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

    ax.set_title(
        "PCA Projection"
    )

    st.pyplot(fig)

# ==================================================
# TSNE VISUALIZATION
# ==================================================

elif page == "t-SNE Visualization":

    st.header("t-SNE Projection")

    processed_data, X = preprocess_data(df)

    perplexity = st.slider(
        "Perplexity",
        5,
        50,
        30
    )

    learning_rate = st.slider(
        "Learning Rate",
        10,
        1000,
        200
    )

    sample_size = st.slider(
        "Sample Size",
        200,
        min(2000, len(X)),
        min(1000, len(X))
    )

    X_sample = X[:sample_size]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=42
    )

    X_tsne = tsne.fit_transform(
        X_sample
    )

    tsne_df = pd.DataFrame(
        {
            "tSNE1": X_tsne[:,0],
            "tSNE2": X_tsne[:,1]
        }
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.scatterplot(
        data=tsne_df,
        x="tSNE1",
        y="tSNE2"
    )

    st.pyplot(fig)

# ==================================================
# PARAMETER ANALYSIS
# ==================================================

elif page == "Parameter Analysis":

    st.header("Understanding Parameters")

    st.markdown("""
### Perplexity

- Controls neighborhood size.
- Typical range: 5–50.
- Low value → local structure.
- High value → global structure.

### Learning Rate

- Controls optimization speed.
- Too low → poor convergence.
- Too high → unstable embedding.

### Random State

- Ensures reproducibility.

### Important

t-SNE is primarily used for visualization and exploration, not prediction.
""")

# ==================================================
# INSIGHTS
# ==================================================

elif page == "Insights":

    st.header("Business Insights")

    st.markdown("""
### Findings

✅ Reduces high-dimensional customer data into 2D.

✅ Preserves local relationships.

✅ Useful for customer behavior exploration.

✅ Helps identify hidden customer segments.

### Applications

- Customer segmentation
- Marketing analysis
- Pattern discovery
- Data visualization

### Advantages

- Excellent visualization
- Reveals hidden structures
- Handles nonlinear relationships

### Limitations

- Computationally expensive
- No prediction capability
- Sensitive to parameters
""")