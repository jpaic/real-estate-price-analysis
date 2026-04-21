import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction & Market Analysis",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("Real Estate Price Prediction & Market Analysis")
st.write(
    "Interactive dashboard for exploratory data analysis and model evaluation "
    "on a US housing dataset."
)

# -----------------------------
# DATA LOAD
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/American_Housing_Data.csv")
    df["Log_Price"] = np.log1p(df["Price"])
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Project Overview")

st.sidebar.write(
    """
This project explores US housing prices using machine learning regression models.
"""
)

st.sidebar.markdown("---")

st.sidebar.header("Navigation")

section = st.sidebar.selectbox(
    "Select view",
    [
        "Dataset Overview",
        "Exploratory Analysis",
        "Model Performance",
        "Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Tech Stack")
st.sidebar.write("""
Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
Streamlit  
""")

st.sidebar.markdown("---")

st.sidebar.subheader("About")
st.sidebar.write("Author : Jovan Paić")

st.sidebar.markdown(
    "[GitHub Repository](https://github.com/jpaic/real-estate-price-analysis)"
)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if section == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# -----------------------------
# EXPLORATORY ANALYSIS
# -----------------------------
elif section == "Exploratory Analysis":

    st.header("Exploratory Data Analysis")

    # -----------------------------
    # DISTRIBUTIONS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Price"], bins=50)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Log Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Log_Price"], bins=50)
        ax.set_xlabel("Log Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # -----------------------------
    # SCATTER
    # -----------------------------
    st.subheader("Income vs Price Relationship")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        df["Median Household Income"],
        df["Log_Price"],
        alpha=0.25,
        s=10
    )
    ax.set_xlabel("Median Household Income")
    ax.set_ylabel("Log Price")
    st.pyplot(fig)

    # -----------------------------
    # CORRELATION
    # -----------------------------
    st.subheader("Feature Correlation")

    corr_features = [
        "Price",
        "Living Space",
        "Beds",
        "Baths",
        "Median Household Income",
        "Zip Code Density",
        "Log_Price"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[corr_features].corr(), cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif section == "Model Performance":

    st.header("Model Evaluation Results")

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree"],
        "R2 Score": [0.78, 0.86],
        "RMSE": [120000, 95000],
        "MAE": [85000, 65000]
    })

    st.subheader("Performance Table")
    st.dataframe(results, use_container_width=True)

    # -----------------------------
    # EQUAL LAYOUT COMPARISON
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(results["Model"], results["R2 Score"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    with col2:
        st.subheader("RMSE Comparison")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(results["Model"], results["RMSE"])
        st.pyplot(fig)

# -----------------------------
# INSIGHTS
# -----------------------------
elif section == "Insights":

    st.header("Key Insights")

    st.markdown(
        """
        - Housing prices are highly right-skewed and benefit from log transformation.
        - Income and living space interactions are strong predictors of price.
        - Geographic and density features significantly affect pricing structure.
        - Decision Tree models capture non-linear relationships more effectively.
        - Feature engineering contributed more than model complexity.
        """
    )

    st.subheader("Conclusion")

    st.write(
        """
        The analysis shows that housing price prediction is driven by complex
        non-linear relationships between socioeconomic and geographic variables.
        Tree-based models outperform linear regression due to their ability to
        capture these interactions.
        """
    )