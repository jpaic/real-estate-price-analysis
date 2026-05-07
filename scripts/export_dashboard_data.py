import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "frontend" / "public" / "dashboard-data.json"


def clean_number(value, digits=4):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    return value


def money(value):
    return f"${float(value):,.0f}"


def histogram(series, bins=36, upper=None):
    values = series.dropna()
    if upper is not None:
        values = values.clip(upper=upper)
    counts, edges = np.histogram(values, bins=bins)
    return [
        {
            "x0": clean_number(edges[i], 2),
            "x1": clean_number(edges[i + 1], 2),
            "label": f"{edges[i]:.2f}-{edges[i + 1]:.2f}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def trendline(points, x_key, y_key):
    xs = np.array([p[x_key] for p in points], dtype=float)
    ys = np.array([p[y_key] for p in points], dtype=float)
    z = np.polyfit(xs, ys, 1)
    x_min, x_max = float(xs.min()), float(xs.max())
    return [
        {"x": clean_number(x_min, 2), "y": clean_number(np.poly1d(z)(x_min), 4)},
        {"x": clean_number(x_max, 2), "y": clean_number(np.poly1d(z)(x_max), 4)},
    ]


def boxplot_groups(df, column, min_value, max_value):
    filtered = df[df[column].between(min_value, max_value)].copy()
    groups = []
    for value in sorted(filtered[column].unique()):
        values = filtered[filtered[column] == value]["Log_Price"]
        groups.append(
            {
                "label": str(int(value) if float(value).is_integer() else value),
                "min": clean_number(values.quantile(0.05), 4),
                "q1": clean_number(values.quantile(0.25), 4),
                "median": clean_number(values.quantile(0.5), 4),
                "q3": clean_number(values.quantile(0.75), 4),
                "max": clean_number(values.quantile(0.95), 4),
            }
        )
    return groups


def load_artifacts():
    lr = joblib.load(ROOT / "models" / "linear_regression.pkl")
    dt = joblib.load(ROOT / "models" / "decision_tree.pkl")
    scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    features = joblib.load(ROOT / "models" / "features.pkl")
    return lr, dt, scaler, features


def load_data(features):
    df = pd.read_csv(ROOT / "data" / "American_Housing_Data.csv")

    df["Median Household Income"] = df["Median Household Income"].fillna(
        df["Median Household Income"].median()
    )
    df = df.drop_duplicates()
    df["Log_Price"] = np.log1p(df["Price"])

    outlier_indices = set()
    for col in ["Living Space", "Beds", "Baths"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 2 * iqr) | (df[col] > q3 + 2 * iqr)
        outlier_indices.update(df[mask].index)
    df = df.drop(index=outlier_indices).reset_index(drop=True)

    state_dummies = pd.get_dummies(df["State"], prefix="State", drop_first=True)
    df = pd.concat([df, state_dummies], axis=1)

    df["Income_x_Space"] = (
        df["Median Household Income"] / 100000
    ) * (df["Living Space"] / 1000)
    df["Density_x_Income"] = (
        df["Zip Code Density"] / 1000
    ) * (df["Median Household Income"] / 100000)
    df["Total_Rooms"] = df["Beds"] + df["Baths"]

    expensive_df = df[df["Price"] >= df["Price"].quantile(0.95)]
    lat_exp = expensive_df["Latitude"].mean()
    lon_exp = expensive_df["Longitude"].mean()
    df["Distance_from_Expensive"] = np.sqrt(
        (df["Latitude"] - lat_exp) ** 2 + (df["Longitude"] - lon_exp) ** 2
    )

    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    return df


def evaluate_models(lr, dt, scaler, features, df):
    x = df[features]
    y = df["Log_Price"]
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    y_test_original = np.expm1(y_test)

    x_test_scaled = scaler.transform(x_test)
    lr_pred_log = lr.predict(x_test_scaled)
    dt_pred_log = dt.predict(x_test)
    y_pred_lr = np.expm1(lr_pred_log)
    y_pred_dt = np.expm1(dt_pred_log)

    lr_m = {
        "r2": round(r2_score(y_test_original, y_pred_lr), 4),
        "rmse": int(np.sqrt(mean_squared_error(y_test_original, y_pred_lr))),
        "mae": int(mean_absolute_error(y_test_original, y_pred_lr)),
        "mape": round(
            np.mean(np.abs((y_test_original - y_pred_lr) / y_test_original)) * 100, 2
        ),
    }
    dt_m = {
        "r2": round(r2_score(y_test_original, y_pred_dt), 4),
        "rmse": int(np.sqrt(mean_squared_error(y_test_original, y_pred_dt))),
        "mae": int(mean_absolute_error(y_test_original, y_pred_dt)),
        "mape": round(
            np.mean(np.abs((y_test_original - y_pred_dt) / y_test_original)) * 100, 2
        ),
        "bestDepth": dt.max_depth,
    }

    fi = (
        pd.Series(dt.feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(15)
    )

    pred_sample = pd.DataFrame(
        {
            "actual": y_test.values,
            "linear": lr_pred_log,
            "tree": dt_pred_log,
        }
    ).sample(min(1000, len(y_test)), random_state=7)

    return lr_m, dt_m, fi, y_test, lr_pred_log, dt_pred_log, pred_sample


def export():
    lr, dt, scaler, features = load_artifacts()
    df = load_data(features)
    lr_m, dt_m, fi, y_test, lr_pred_log, dt_pred_log, pred_sample = evaluate_models(
        lr, dt, scaler, features, df
    )

    preview_cols = [
        "Price",
        "Living Space",
        "Beds",
        "Baths",
        "Median Household Income",
        "Zip Code Density",
        "State",
    ]
    stats_cols = [
        "Price",
        "Living Space",
        "Beds",
        "Baths",
        "Median Household Income",
        "Zip Code Density",
    ]
    desc = df["Price"].describe()
    summary = df[stats_cols].describe().T.round(2)

    income_sample = (
        df.sample(min(900, len(df)), random_state=1)[
            ["Median Household Income", "Log_Price"]
        ]
        .rename(columns={"Median Household Income": "x", "Log_Price": "y"})
        .round(4)
        .to_dict("records")
    )
    space_sample = (
        df[df["Living Space"] < df["Living Space"].quantile(0.98)]
        .sample(min(900, len(df)), random_state=2)[["Living Space", "Log_Price"]]
        .rename(columns={"Living Space": "x", "Log_Price": "y"})
        .round(4)
        .to_dict("records")
    )

    corr_cols = [
        "Price",
        "Log_Price",
        "Living Space",
        "Beds",
        "Baths",
        "Median Household Income",
        "Zip Code Density",
        "Income_x_Space",
        "Total_Rooms",
    ]
    corr = df[corr_cols].corr().round(2)

    payload = {
        "overviewMetrics": [
            {
                "label": "Total records",
                "value": f"{df.shape[0]:,}",
                "sub": "housing listings",
            },
            {
                "label": "Features",
                "value": str(df.shape[1]),
                "sub": "raw + engineered columns",
            },
            {
                "label": "Missing values",
                "value": str(int(df.isnull().sum().sum())),
                "sub": "clean dataset",
            },
            {
                "label": "States covered",
                "value": str(df["State"].nunique()),
                "sub": "across the US",
            },
        ],
        "dataPreview": df[preview_cols].head(10).to_dict("records"),
        "priceStatistics": [
            {"Statistic": "Count", "Price ($)": f"{desc['count']:,.0f}"},
            {"Statistic": "Mean", "Price ($)": money(desc["mean"])},
            {"Statistic": "Std Dev", "Price ($)": money(desc["std"])},
            {"Statistic": "Min", "Price ($)": money(desc["min"])},
            {"Statistic": "25th pct", "Price ($)": money(desc["25%"])},
            {"Statistic": "Median", "Price ($)": money(desc["50%"])},
            {"Statistic": "75th pct", "Price ($)": money(desc["75%"])},
            {"Statistic": "Max", "Price ($)": money(desc["max"])},
        ],
        "statisticalSummary": [
            {
                "Feature": index,
                **{col: clean_number(value, 2) for col, value in row.items()},
            }
            for index, row in summary.iterrows()
        ],
        "charts": {
            "priceDistribution": histogram(
                df["Price"], bins=42, upper=df["Price"].quantile(0.99)
            ),
            "logPriceDistribution": histogram(df["Log_Price"], bins=42),
            "incomeScatter": income_sample,
            "incomeTrend": trendline(income_sample, "x", "y"),
            "spaceScatter": space_sample,
            "spaceTrend": trendline(space_sample, "x", "y"),
            "bedBoxes": boxplot_groups(df, "Beds", 1, 6),
            "bathBoxes": boxplot_groups(df, "Baths", 1, 5),
            "correlationMatrix": {
                "columns": corr_cols,
                "values": [
                    [clean_number(value, 2) for value in row]
                    for row in corr.values.tolist()
                ],
            },
            "predictionSample": pred_sample.round(4).to_dict("records"),
            "linearResiduals": histogram(
                pd.Series(y_test.values - lr_pred_log), bins=42
            ),
            "treeResiduals": histogram(pd.Series(y_test.values - dt_pred_log), bins=42),
            "featureImportances": [
                {"feature": index, "importance": clean_number(value, 5)}
                for index, value in fi.sort_values(ascending=True).items()
            ],
        },
        "modelMetrics": {
            "linear": lr_m,
            "tree": dt_m,
            "rmseReduction": round((1 - dt_m["rmse"] / lr_m["rmse"]) * 100),
        },
        "meta": {
            "author": "Jovan Paić",
            "repository": "https://github.com/jpaic/real-estate-price-analysis",
            "techStack": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "Next.js",
                "TypeScript",
                "Vercel",
            ],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    export()
