# Real Estate Price Prediction & Market Analysis

End-to-end machine learning project for predicting and analyzing U.S. housing prices using regression models, feature engineering, and an interactive Next.js dashboard.

## Live Demo

[Real Estate Price Analysis Dashboard](https://github.com/jpaic/real-estate-price-analysis)

The Next.js dashboard is the main serving/presentation layer. It displays precomputed model results, visualizations, and dataset insights from static JSON, and is optimized for Vercel deployment.

## Key Highlights

- Built and compared Linear Regression and Decision Tree models
- Applied feature engineering to capture socioeconomic and spatial relationships
- Improved model stability with log transformation of the target variable
- Evaluated models using RMSE, MAE, MAPE, and R2
- Migrated the original Streamlit dashboard to a Vercel-ready Next.js app

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Next.js
- TypeScript
- Vercel

## Objective

To predict housing prices using supervised machine learning and analyze how linear and non-linear models perform on structured real estate data with complex feature interactions.

## Dataset

American Housing Dataset (CSV)

The dataset includes property-level and regional features such as:

- Living area size
- Number of bedrooms and bathrooms
- Median household income
- Population density at ZIP level
- Geographic coordinates
- State information

Target variable:

- House Price, log-transformed as `Log_Price` during training

## Approach

### Data Preprocessing

- Missing values handled using median imputation
- Duplicate rows removed
- Outliers treated using IQR filtering
- Log transformation applied to reduce skewness in the target variable
- One-hot encoding applied to categorical state values

### Feature Engineering

- Income x Living Space interaction
- Density x Income interaction
- Total Rooms
- Distance from high-price cluster centroid

### Modeling

- Linear Regression baseline
- Decision Tree Regressor for non-linear relationships

## Dashboard

The Next.js dashboard includes:

- Dataset overview and cleaned-data metrics
- Data preview, price statistics, and statistical summaries
- Exploratory charts for price, income, living space, beds, and baths
- Correlation matrix
- Linear Regression vs Decision Tree model cards
- Actual vs predicted plots
- Residual distributions
- Decision Tree feature importances
- Key findings and conclusion

## Dashboard Data

The deployed frontend reads precomputed data from:

```text
app/frontend/public/dashboard-data.json
```

Regenerate it from the repo root with:

```bash
python scripts/export_dashboard_data.py
```

The script loads the dataset and trained model files, reruns the same preprocessing and model-evaluation calculations, and writes the JSON consumed by the Next.js app. This keeps the deployed site simple because Vercel does not need to run Python, pandas, or scikit-learn at request time.

## Results

- Log transformation improved model stability and reduced skew impact
- Feature engineering improved predictive performance
- Decision Tree outperformed Linear Regression across the tracked metrics
- Linear Regression struggled with non-linear and spatial relationships in the data

## Conclusion

Housing prices are influenced by complex, non-linear relationships between geography, income, and property features. The Decision Tree model captured these interactions more effectively than Linear Regression.

Feature engineering improved overall performance, but the final results remain moderately accurate, reflecting the inherent difficulty of predicting housing prices from structured tabular data alone. Future improvements could include Random Forest, Gradient Boosting, richer geospatial features, and cross-validation-based hyperparameter tuning.
