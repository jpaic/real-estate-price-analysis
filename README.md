# Real Estate Price Prediction & Market Analysis (Machine Learning Regression)

## Overview
This project applies supervised machine learning techniques to predict US housing prices based on demographic, geographic, and structural features. The goal is to compare a simple linear model against a non-linear tree-based model and analyze which better captures complex relationships in housing data.

---

## Objective
- Predict housing prices using regression models
- Improve prediction performance through feature engineering
- Compare linear vs non-linear approaches
- Evaluate models using real-dollar error metrics

---

## Dataset
**American Housing Dataset (CSV)**  
The dataset contains property-level and regional features such as:
- Living space
- Number of bedrooms and bathrooms
- Median household income
- Zip code density
- Geographic coordinates (latitude/longitude)
- State information

The target variable is:
- **Price** (log-transformed during modeling as `Log_Price`)

---

## Data Preprocessing
- Handled missing values using median imputation
- Removed duplicate rows
- Treated outliers using IQR-based filtering
- Applied log transformation to reduce skewness in price distribution
- One-hot encoded categorical variables (State)

---

## Feature Engineering
- Income × Living Space interaction
- Density × Income interaction
- Total Rooms (Beds + Baths)
- Distance from high-price cluster centroid

These features capture non-linear socioeconomic and spatial relationships.

---

## Models Used

### 1. Linear Regression
- Baseline model
- Trained on standardized features
- Assumes linear relationships

### 2. Decision Tree Regressor
- Captures non-linear interactions
- Tuned using different max depths
- Selected based on best test R² score

---

## Evaluation Metrics
- R² Score
- RMSE (in dollars)
- MAE (in dollars)
- MAPE (% error)

All metrics are evaluated on the original price scale.

---

## Key Results
- Log transformation improved model stability
- Engineered features were strong predictors
- Decision Tree outperformed Linear Regression
- Linear model struggled with non-linear interactions

---

## Streamlit Dashboard
An interactive dashboard was built to visualize the dataset and model performance.
 
[Streamlit App](https://jpaic-real-estate-analysis.streamlit.app)

---

## Conclusion
Housing prices are driven by complex non-linear relationships between income, geography, and property structure. Tree-based models better capture these interactions compared to linear regression.

Feature engineering played a major role in improving predictive performance.

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
