# Real Estate Price Prediction & Market Analysis

End-to-end machine learning project for predicting U.S. housing prices using regression models and feature engineering. Developed as a university project focused on supervised learning for regression tasks.

---

## Live Demo
[Streamlit App](https://jpaic-real-estate-analysis.streamlit.app)

---

## Key Highlights
- Built and compared Linear Regression and Decision Tree models  
- Applied feature engineering to capture socio-economic and spatial relationships  
- Improved performance using log transformation of the target variable  
- Evaluated models using real-world error metrics (RMSE, MAE, MAPE)  
- Full ML pipeline: preprocessing, modeling, evaluation, and deployment  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Joblib  
- Streamlit  

---

## Objective
To predict housing prices using supervised machine learning and analyze how linear and non-linear models perform on structured real estate data with complex feature interactions.

---

## Dataset
American Housing Dataset (CSV)

The dataset includes property-level and regional features such as:
- Living area size  
- Number of bedrooms and bathrooms  
- Median household income  
- Population density (ZIP-level)  
- Geographic coordinates (latitude and longitude)  
- State information  

**Target variable:**
- House Price (log-transformed as `Log_Price` during training)

---

## Approach

### Data Preprocessing
- Missing values handled using median imputation  
- Duplicate rows removed  
- Outliers treated using IQR filtering  
- Log transformation applied to reduce skewness in target distribution  
- One-hot encoding applied to categorical variables (State)  

### Feature Engineering
- Income × Living Space interaction  
- Density × Income interaction  
- Total Rooms (Bedrooms + Bathrooms)  
- Distance from high-price cluster centroid  

### Modeling
- Linear Regression (baseline, assumes linear relationships)  
- Decision Tree Regressor (captures non-linear interactions, tuned via depth)  

---

## Results
- Log transformation improved model stability and reduced skew impact  
- Feature engineering significantly improved predictive performance  
- Decision Tree outperformed Linear Regression across all evaluation metrics  
- Linear model struggled with non-linear and spatial relationships in the data  

---

## Conclusion
Housing prices are influenced by complex, non-linear relationships between geography, income, and property features. The Decision Tree model captured these interactions more effectively than Linear Regression.

Feature engineering improved overall performance, but the final results remain moderately accurate (R² ≈ 0.65), with relatively high error in absolute price terms. This reflects the inherent difficulty of predicting housing prices using structured tabular data with limited model complexity.

In this context, ensemble methods such as Random Forest or Gradient Boosting would likely provide better performance, as they are more robust to variance, overfitting, and complex feature interactions compared to single decision trees.