import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME / GLOBAL CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #12121f !important;
}
[data-testid="stSidebar"] * {
    color: rgba(210, 210, 235, 0.75) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e0e0f5 !important;
}
[data-testid="stSidebar"] .stSelectbox label {
    color: rgba(210,210,235,0.5) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #c8c8e8 !important;
}

/* ---------- main area ---------- */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

.section-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background: rgba(91,106,240,0.12);
    color: #5b6af0;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 8px;
    margin-top: 4px;
    line-height: 1.6;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin-bottom: 1.4rem;
    margin-top: 2px;
    line-height: 1.25;
}

[data-theme="light"] .section-title {
    color: #111122;
}

[data-theme="dark"] .section-title {
    color: #ffffff;
}
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: #fff;
    border: 0.5px solid rgba(0,0,0,0.08);
    border-radius: 10px;
    padding: 14px 18px;
}
.metric-card .m-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 6px;
}
.metric-card .m-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #111122;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: 0.72rem;
    color: #aaa;
    margin-top: 3px;
}

.model-grid { display: flex; gap: 14px; margin-bottom: 1rem; }
.model-card {
    flex: 1;
    background: #fff;
    border: 0.5px solid rgba(0,0,0,0.08);
    border-radius: 10px;
    padding: 16px 18px;
}
.model-card.best { border-color: #1d9e75; border-width: 1.5px; }
.model-card .mc-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #111122;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
}
.badge-green { background: rgba(29,158,117,0.1); color: #0f6e56; }
.badge-blue  { background: rgba(91,106,240,0.1);  color: #3a47b8; }
.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    padding: 6px 0;
    border-top: 0.5px solid rgba(0,0,0,0.06);
    color: #555;
}
.stat-row .sv { font-weight: 600; color: #111122; }
.bar-wrap { margin-top: 12px; }
.bar-label { font-size: 0.68rem; color: #999; margin-bottom: 4px; }
.bar-track {
    height: 6px;
    background: #f0f0f5;
    border-radius: 20px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ── PALETTE ────────────────────────────────────────────────────────────────────
INDIGO = "#5b6af0"
TEAL   = "#1d9e75"
AMBER  = "#ef9f27"

def chart_style(ax, xlabel="", ylabel=""):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e0e0ec")
    ax.tick_params(colors="#999", labelsize=9)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#f0f0f5", linewidth=0.7)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#888")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#888")

# ── LOAD ARTIFACTS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    lr       = joblib.load("models/linear_regression.pkl")
    dt       = joblib.load("models/decision_tree.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    return lr, dt, scaler, features

# ── LOAD & PREPARE DATA ────────────────────────────────────────────────────────
@st.cache_data
def load_data(features):
    df = pd.read_csv("data/American_Housing_Data.csv")

    df["Median Household Income"] = df["Median Household Income"].fillna(
        df["Median Household Income"].median()
    )
    df = df.drop_duplicates()
    df["Log_Price"] = np.log1p(df["Price"])

    key_columns = ["Living Space", "Beds", "Baths"]
    outlier_indices = set()
    for col in key_columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 2 * IQR) | (df[col] > Q3 + 2 * IQR)
        outlier_indices.update(df[mask].index)
    df = df.drop(index=outlier_indices).reset_index(drop=True)

    state_dummies = pd.get_dummies(df["State"], prefix="State", drop_first=True)
    df = pd.concat([df, state_dummies], axis=1)

    df["Income_x_Space"]   = (df["Median Household Income"] / 100000) * (df["Living Space"] / 1000)
    df["Density_x_Income"] = (df["Zip Code Density"] / 1000) * (df["Median Household Income"] / 100000)
    df["Total_Rooms"]      = df["Beds"] + df["Baths"]

    expensive_df = df[df["Price"] >= df["Price"].quantile(0.95)]
    lat_exp = expensive_df["Latitude"].mean()
    lon_exp = expensive_df["Longitude"].mean()
    df["Distance_from_Expensive"] = np.sqrt(
        (df["Latitude"] - lat_exp) ** 2 + (df["Longitude"] - lon_exp) ** 2
    )

    return df

# ── EVALUATE MODELS ON HELD-OUT TEST SET ──────────────────────────────────────
@st.cache_data
def evaluate_models(_lr, _dt, _scaler, features, _df):
    from sklearn.model_selection import train_test_split

    X = _df[features]
    y = _df["Log_Price"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_test_original = np.expm1(y_test)

    X_test_scaled = _scaler.transform(X_test)

    # Linear Regression
    lr_pred_log = _lr.predict(X_test_scaled)
    y_pred_lr   = np.expm1(lr_pred_log)

    lr_m = {
        "R2":   round(r2_score(y_test_original, y_pred_lr), 4),
        "RMSE": int(np.sqrt(mean_squared_error(y_test_original, y_pred_lr))),
        "MAE":  int(mean_absolute_error(y_test_original, y_pred_lr)),
        "MAPE": round(np.mean(np.abs((y_test_original - y_pred_lr) / y_test_original)) * 100, 2),
    }

    # Decision Tree
    dt_pred_log = _dt.predict(X_test)
    y_pred_dt   = np.expm1(dt_pred_log)

    dt_m = {
        "R2":         round(r2_score(y_test_original, y_pred_dt), 4),
        "RMSE":       int(np.sqrt(mean_squared_error(y_test_original, y_pred_dt))),
        "MAE":        int(mean_absolute_error(y_test_original, y_pred_dt)),
        "MAPE":       round(np.mean(np.abs((y_test_original - y_pred_dt) / y_test_original)) * 100, 2),
        "best_depth": _dt.max_depth,
    }

    # Feature importances
    fi = pd.Series(_dt.feature_importances_, index=features).sort_values(ascending=True)

    return lr_m, dt_m, fi, y_test, lr_pred_log, dt_pred_log

# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
lr, dt, scaler, features = load_artifacts()
df = load_data(features)
lr_m, dt_m, fi, y_test, lr_pred_log, dt_pred_log = evaluate_models(
    lr, dt, scaler, features, df
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
      <div style="width:32px;height:32px;background:#5b6af0;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  font-weight:700;font-size:14px;color:#fff;">RE</div>
      <div>
        <div style="font-size:0.85rem;font-weight:600;color:#e0e0f5 !important;">Real Estate</div>
        <div style="font-size:0.7rem;color:rgba(210,210,235,0.45) !important;">Price Analysis</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    section = st.selectbox(
        "NAVIGATION",
        ["Dataset Overview", "Exploratory Analysis", "Model Performance", "Insights"],
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;
                color:rgba(210,210,235,0.35);margin-bottom:8px;">Tech Stack</div>
    <div style="display:flex;flex-wrap:wrap;gap:5px;">
      """ + "".join(
        f'<span style="font-size:0.68rem;background:rgba(255,255,255,0.07);'
        f'color:rgba(210,210,235,0.55);padding:2px 8px;border-radius:20px;">{t}</span>'
        for t in ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "Streamlit"]
    ) + """
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(210,210,235,0.35);">Author</div>
    <div style="font-size:0.82rem;color:rgba(210,210,235,0.65);margin-bottom:6px;">Jovan Paić</div>
    <a href="https://github.com/jpaic/real-estate-price-analysis"
       style="font-size:0.75rem;color:#8090e0;text-decoration:none;">
      ↗ GitHub Repository
    </a>
    """, unsafe_allow_html=True)


# ── DATASET OVERVIEW ───────────────────────────────────────────────────────────
if section == "Dataset Overview":

    st.markdown('<div class="section-badge">01 — Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">American Housing Dataset</div>', unsafe_allow_html=True)

    missing = int(df.isnull().sum().sum())
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="m-label">Total records</div>
        <div class="m-value">{df.shape[0]:,}</div>
        <div class="m-sub">housing listings</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Features</div>
        <div class="m-value">{df.shape[1]}</div>
        <div class="m-sub">raw + engineered columns</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Missing values</div>
        <div class="m-value">{missing}</div>
        <div class="m-sub">{"clean dataset" if missing == 0 else "imputed with median"}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">States covered</div>
        <div class="m-value">{df["State"].nunique() if "State" in df.columns else "—"}</div>
        <div class="m-sub">across the US</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.75rem; color:rgba(100,100,100,0.75); font-style:italic; margin-top:-6px;'>"
        "Note: these values reflect the dataset after preprocessing, outlier removal, and feature engineering."
        "</div>",
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### Data preview")
        st.dataframe(
            df[["Price", "Living Space", "Beds", "Baths",
                "Median Household Income", "Zip Code Density", "State"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("##### Price statistics")
        desc = df["Price"].describe()
        desc_df = pd.DataFrame({
            "Statistic": ["Count", "Mean", "Std Dev", "Min", "25th pct", "Median", "75th pct", "Max"],
            "Price ($)": [
                f"{desc['count']:,.0f}",
                f"${desc['mean']:,.0f}",
                f"${desc['std']:,.0f}",
                f"${desc['min']:,.0f}",
                f"${desc['25%']:,.0f}",
                f"${desc['50%']:,.0f}",
                f"${desc['75%']:,.0f}",
                f"${desc['max']:,.0f}",
            ]
        })
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.markdown("##### Full statistical summary")
    st.dataframe(
        df[["Price", "Living Space", "Beds", "Baths",
            "Median Household Income", "Zip Code Density"]].describe().T.round(2),
        use_container_width=True,
    )


# ── EXPLORATORY ANALYSIS ───────────────────────────────────────────────────────
elif section == "Exploratory Analysis":

    st.markdown('<div class="section-badge">02 — EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Price distribution (raw)")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(df["Price"].clip(upper=df["Price"].quantile(0.99)),
                bins=60, color=INDIGO, alpha=0.85, edgecolor="white", linewidth=0.3)
        chart_style(ax, "Price ($)", "Frequency")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("##### Log price distribution (transformed)")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(df["Log_Price"], bins=60, color=TEAL, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        chart_style(ax, "log(1 + Price)", "Frequency")
        fig.tight_layout()
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Median household income vs log price")
        sample = df.sample(min(5000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(sample["Median Household Income"], sample["Log_Price"],
                   alpha=0.18, s=8, c=INDIGO)
        z = np.polyfit(sample["Median Household Income"], sample["Log_Price"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sample["Median Household Income"].min(),
                         sample["Median Household Income"].max(), 200)
        ax.plot(xs, p(xs), color=AMBER, linewidth=1.8, label="trend")
        ax.legend(fontsize=8)
        chart_style(ax, "Median Household Income ($)", "Log Price")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("##### Living space vs log price")
        sample2 = df[df["Living Space"] < df["Living Space"].quantile(0.98)].sample(
            min(4000, len(df)), random_state=2
        )
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(sample2["Living Space"], sample2["Log_Price"],
                   alpha=0.18, s=8, c=TEAL)
        z2 = np.polyfit(sample2["Living Space"], sample2["Log_Price"], 1)
        p2 = np.poly1d(z2)
        xs2 = np.linspace(sample2["Living Space"].min(), sample2["Living Space"].max(), 200)
        ax.plot(xs2, p2(xs2), color=AMBER, linewidth=1.8, label="trend")
        ax.legend(fontsize=8)
        chart_style(ax, "Living Space (sq ft)", "Log Price")
        fig.tight_layout()
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Price by number of bedrooms")
        bed_data = df[df["Beds"].between(1, 6)].copy()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        groups = [bed_data[bed_data["Beds"] == b]["Log_Price"].values
                  for b in sorted(bed_data["Beds"].unique())]
        bp = ax.boxplot(groups, patch_artist=True, notch=False, widths=0.5,
                        medianprops=dict(color=AMBER, linewidth=2),
                        whiskerprops=dict(color="#ccc"),
                        capprops=dict(color="#ccc"),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#bbb"))
        for patch in bp["boxes"]:
            patch.set_facecolor(INDIGO)
            patch.set_alpha(0.7)
        ax.set_xticklabels(sorted(bed_data["Beds"].unique()))
        chart_style(ax, "Bedrooms", "Log Price")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("##### Price by number of bathrooms")
        bath_data = df[df["Baths"].between(1, 5)].copy()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        groups2 = [bath_data[bath_data["Baths"] == b]["Log_Price"].values
                   for b in sorted(bath_data["Baths"].unique())]
        bp2 = ax.boxplot(groups2, patch_artist=True, notch=False, widths=0.5,
                         medianprops=dict(color=AMBER, linewidth=2),
                         whiskerprops=dict(color="#ccc"),
                         capprops=dict(color="#ccc"),
                         flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#bbb"))
        for patch in bp2["boxes"]:
            patch.set_facecolor(TEAL)
            patch.set_alpha(0.7)
        ax.set_xticklabels(sorted(bath_data["Baths"].unique()))
        chart_style(ax, "Bathrooms", "Log Price")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("##### Feature correlation matrix")
    corr_cols = [
        "Price", "Log_Price", "Living Space", "Beds", "Baths",
        "Median Household Income", "Zip Code Density",
        "Income_x_Space", "Total_Rooms",
    ]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.4, linecolor="#f0f0f5",
        annot_kws={"size": 8}, ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    st.pyplot(fig)


# ── MODEL PERFORMANCE ──────────────────────────────────────────────────────────
elif section == "Model Performance":

    st.markdown('<div class="section-badge">03 — Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Evaluation Results</div>', unsafe_allow_html=True)

    lr_r2_pct = int(lr_m["R2"] * 100)
    dt_r2_pct = int(dt_m["R2"] * 100)

    st.markdown(f"""
    <div class="model-grid">
      <div class="model-card">
        <div class="mc-name">
          Linear Regression
          <span class="badge badge-blue">Baseline</span>
        </div>
        <div class="stat-row"><span>R² Score</span><span class="sv">{lr_m["R2"]}</span></div>
        <div class="stat-row"><span>RMSE</span><span class="sv">${lr_m["RMSE"]:,}</span></div>
        <div class="stat-row"><span>MAE</span><span class="sv">${lr_m["MAE"]:,}</span></div>
        <div class="stat-row"><span>MAPE</span><span class="sv">{lr_m["MAPE"]}%</span></div>
        <div class="bar-wrap">
          <div class="bar-label">R² score ({lr_r2_pct}%)</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{lr_r2_pct}%;background:#5b6af0;"></div>
          </div>
        </div>
      </div>
      <div class="model-card best">
        <div class="mc-name">
          Decision Tree Regressor
          <span class="badge badge-green">Best model · depth {dt_m["best_depth"]}</span>
        </div>
        <div class="stat-row"><span>R² Score</span><span class="sv" style="color:#1d9e75;">{dt_m["R2"]}</span></div>
        <div class="stat-row"><span>RMSE</span><span class="sv">${dt_m["RMSE"]:,}</span></div>
        <div class="stat-row"><span>MAE</span><span class="sv">${dt_m["MAE"]:,}</span></div>
        <div class="stat-row"><span>MAPE</span><span class="sv">{dt_m["MAPE"]}%</span></div>
        <div class="bar-wrap">
          <div class="bar-label">R² score ({dt_r2_pct}%)</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{dt_r2_pct}%;background:#1d9e75;"></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Actual vs predicted — Linear Regression")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, lr_pred_log, alpha=0.25, s=7, color=INDIGO)
        lims = [min(y_test.min(), lr_pred_log.min()), max(y_test.max(), lr_pred_log.max())]
        ax.plot(lims, lims, "--", color=AMBER, linewidth=1.4, label="perfect fit")
        ax.legend(fontsize=8)
        chart_style(ax, "Actual log price", "Predicted log price")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("##### Actual vs predicted — Decision Tree")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, dt_pred_log, alpha=0.25, s=7, color=TEAL)
        lims2 = [min(y_test.min(), dt_pred_log.min()), max(y_test.max(), dt_pred_log.max())]
        ax.plot(lims2, lims2, "--", color=AMBER, linewidth=1.4, label="perfect fit")
        ax.legend(fontsize=8)
        chart_style(ax, "Actual log price", "Predicted log price")
        fig.tight_layout()
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Residuals — Linear Regression")
        res_lr = y_test.values - lr_pred_log
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(res_lr, bins=60, color=INDIGO, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=AMBER, linewidth=1.5, linestyle="--")
        chart_style(ax, "Residual (log scale)", "Frequency")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("##### Residuals — Decision Tree")
        res_dt = y_test.values - dt_pred_log
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(res_dt, bins=60, color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=AMBER, linewidth=1.5, linestyle="--")
        chart_style(ax, "Residual (log scale)", "Frequency")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("##### Feature importances (Decision Tree)")
    fi_top = fi.tail(15)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [TEAL if v == fi.max() else INDIGO for v in fi_top.values]
    ax.barh(fi_top.index, fi_top.values, color=colors, alpha=0.85, height=0.6)
    chart_style(ax, "Importance score", "")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("##### Side-by-side metric comparison")
    results_df = pd.DataFrame({
        "Metric":            ["R² Score", "RMSE ($)", "MAE ($)", "MAPE (%)"],
        "Linear Regression": [lr_m["R2"], f"${lr_m['RMSE']:,}", f"${lr_m['MAE']:,}", f"{lr_m['MAPE']}%"],
        "Decision Tree":     [dt_m["R2"], f"${dt_m['RMSE']:,}", f"${dt_m['MAE']:,}", f"{dt_m['MAPE']}%"],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)


# ── INSIGHTS ───────────────────────────────────────────────────────────────────
elif section == "Insights":

    st.markdown('<div class="section-badge">04 — Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)

    st.markdown("#### Data & Target Variable")
    st.markdown(
        "Housing prices in the dataset are heavily right-skewed (skewness ≈ 9.7). "
        "Applying a log transformation via log1p brought the distribution close to normal "
        "(skewness ≈ 0.17), which significantly stabilised model training and improved metric reliability."
    )

    st.markdown("#### Feature Importance")
    st.markdown(
        "The engineered **Income × Living Space** interaction term ranked as the single most important "
        "feature in the Decision Tree, ahead of either variable on its own. This confirms that "
        "socioeconomic standing and physical size compound non-linearly in pricing. "
        "**Zip code density** contributed meaningfully even after income was controlled for, "
        "reflecting urban vs. suburban price premiums. "
        "**Distance from the expensive-home centroid** — a geospatial feature derived from the "
        "top 5% priciest listings — also ranked highly, capturing neighbourhood prestige effects."
    )

    st.markdown("#### Model Comparison")
    st.markdown(
        f"The Decision Tree Regressor (optimal depth: **{dt_m['best_depth']}**) outperformed "
        f"Linear Regression across every metric. "
        f"R² improved from **{lr_m['R2']}** to **{dt_m['R2']}**, and RMSE dropped from "
        f"**\\${lr_m['RMSE']:,}** to **\\${dt_m['RMSE']:,}** — roughly a "
        f"**{round((1 - dt_m['RMSE']/lr_m['RMSE'])*100)}% reduction**. "
        "This gap exists because housing prices are shaped by non-linear interactions between "
        "geography, income, and structural variables that a linear model cannot capture."
    )

    st.markdown("#### Feature Engineering Impact")
    st.markdown(
        "Running the same Decision Tree without the engineered features (using only raw columns) "
        "produced a noticeably lower R². The performance gains came primarily from the interaction "
        "terms and geospatial distance feature — not from model complexity alone. "
        "This highlights that thoughtful feature construction can matter more than algorithm choice."
    )

    st.divider()

    st.markdown("#### Conclusion")
    st.markdown(
        "US housing prices are determined by a complex interplay of socioeconomic, geographic, and "
        "structural variables. Tree-based models capture these interactions more faithfully than "
        "linear approaches. Future work could explore **gradient-boosted ensembles** (XGBoost, LightGBM), "
        "**cross-validation** with GridSearchCV for hyperparameter tuning, and richer "
        "**geospatial features** derived from latitude/longitude to further improve predictive accuracy."
    )