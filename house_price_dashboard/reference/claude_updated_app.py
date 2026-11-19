"""
app/app.py
----------
House Price Intelligence Dashboard — King County, WA House Sales
Streamlit + Plotly · All structure, layout and features preserved from original.
Run: streamlit run app/app.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ── resolve paths regardless of CWD ──────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DATA_PATH  = os.path.join(ROOT_DIR, "data", "house_prices.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")

# ── Plotly import guard ───────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Intelligence Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3d4270; border-radius: 12px;
        padding: 20px 24px; text-align: center; margin-bottom: 12px;
    }
    .kpi-label { color: #8b93c4; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .kpi-value { color: #ffffff; font-size: 1.9rem; font-weight: 700; }
    .kpi-sub   { color: #6be3b0; font-size: 0.78rem; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #c5cef0;
        border-left: 3px solid #5b6fd6; padding-left: 10px;
        margin: 24px 0 12px 0;
    }
    .metric-pill {
        display: inline-block; background: #1e2130; border: 1px solid #3d4270;
        border-radius: 20px; padding: 4px 14px; font-size: 0.82rem; color: #c5cef0; margin: 3px;
    }
    .stDataFrame { border-radius: 8px; }
    section[data-testid="stSidebar"] { background: #13162a; border-right: 1px solid #2a2f52; }
    .sidebar-title { color: #8b93c4; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe Plotly chart renderer
# ─────────────────────────────────────────────────────────────────────────────
def show_chart(fig, key=None):
    if PLOTLY_AVAILABLE:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.warning("Plotly not installed. Run: pip install plotly")

CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1d2e",
    plot_bgcolor="#1a1d2e",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

# ── NEW: introduce controlled dirt into a copy of the raw data ────────────────
@st.cache_data
def make_dirty(seed: int = 42) -> pd.DataFrame:
    """
    Returns a COPY of the raw dataset with controlled issues injected:
      - Missing values in 3–5 % of rows across 5 key columns
      - 50 duplicate rows appended
    The original CSV file is never touched.
    """
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(DATA_PATH)
    n   = len(df)

    # Columns to corrupt and their target missing-value fractions
    corrupt = {
        "Price":        0.03,   # 3 %
        "Sqft_living":  0.04,   # 4 %
        "Grade":        0.02,   # 2 %
        "Bathrooms":    0.03,   # 3 %
        "Yr_built":     0.02,   # 2 %
    }
    for col, frac in corrupt.items():
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Append 50 duplicate rows (sampled from the original)
    dup_rows = df.dropna().sample(n=50, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_clean():
    # ── START FROM THE DIRTY COPY, not the raw CSV ───────────────────────────
    df = make_dirty()

    # ── record before state ──────────────────────────────────────────────────
    before_rows    = len(df)
    before_dtypes  = df.dtypes.copy()
    missing_before = df.isnull().sum()

    # ── remove duplicates ────────────────────────────────────────────────────
    dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    # ── parse & engineer features ────────────────────────────────────────────
    df["Sale_Year"]    = df["Date"].str[:4].astype(int)
    df["Sale_Month"]   = df["Date"].str[4:6].astype(int)
    df["Has_Basement"] = (df["Sqft_basement"] > 0).astype(int)
    df["Renovated"]    = (df["Yr_renovated"] > 0).astype(int)
    df["Age"]          = df["Sale_Year"] - df["Yr_built"]

    # ── fix invalid ages ─────────────────────────────────────────────────────
    df.loc[df["Age"] < 0, "Age"] = 0

    # ── fix outlier bedrooms (33 bedrooms is likely a data error) ─────────────
    df = df[df["Bedrooms"] <= 15]

    # ── fix zero-price rows if any ────────────────────────────────────────────
    df = df[df["Price"] > 0]

    # ── no missing values in this dataset, but fill numerics just in case ─────
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    after_rows    = len(df)
    missing_after = df.isnull().sum()

    meta = {
        "before_rows":    before_rows,
        "after_rows":     after_rows,
        "dups_removed":   dups,
        "missing_before": missing_before,
        "missing_after":  missing_after,
        "before_dtypes":  before_dtypes,
    }
    return df, meta

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

raw_df         = load_raw()
clean_df, meta = load_clean()
bundle         = load_model()

lr_pipeline   = bundle["lr_pipeline"]
rf_pipeline   = bundle["rf_pipeline"]
lr_metrics    = bundle["lr_metrics"]
rf_metrics    = bundle["rf_metrics"]
NUM_FEATURES  = bundle["num_features"]
CAT_FEATURES  = bundle["cat_features"]
ALL_FEATURES  = bundle["all_features"]
importance_df = bundle["importance_df"]
y_test        = bundle["y_test"]
y_pred_lr     = bundle["y_pred_lr"]
y_pred_rf     = bundle["y_pred_rf"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price IQ")
    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    page = st.radio(
        "Navigate",
        ["📊 Overview & KPIs",
         "🧹 Data Quality",
         "📈 Visualizations",
         "📐 Regression Lab",
         "🤖 ML Models",
         "🔮 Predict Price"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Prediction Inputs ─────────────────────────────────────────────────────
    st.markdown("### 🔮 Quick Predict")
    st.caption("Fill inputs then go to Predict Price tab")

    sqft_living  = st.slider("Living Area (sq ft)",
                              int(clean_df.Sqft_living.min()),
                              int(clean_df.Sqft_living.max()), 2000)
    sqft_lot     = st.slider("Lot Size (sq ft)", 500,
                              int(clean_df.Sqft_lot.quantile(0.99)), 8000)
    sqft_above   = st.slider("Sqft Above Basement", 300,
                              int(clean_df.Sqft_above.max()), 1500)
    sqft_bsmt    = st.slider("Basement Sq ft", 0,
                              int(clean_df.Sqft_basement.max()), 500)
    grade        = st.slider("Grade (1–13)", 1, 13, 7)
    condition    = st.slider("Condition (1–5)", 1, 5, 3)
    bedrooms     = st.slider("Bedrooms", 1, 10, 3)
    bathrooms    = st.slider("Bathrooms", 1.0, 6.0, 2.0, step=0.25)
    floors       = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    view         = st.slider("View (0–4)", 0, 4, 0)
    waterfront   = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")
    yr_built     = st.slider("Year Built", int(clean_df.Yr_built.min()),
                              int(clean_df.Yr_built.max()), 1990)
    renovated    = st.selectbox("Ever Renovated", [0, 1], format_func=lambda x: "Yes" if x else "No")
    sqft_l15     = st.slider("Neighbors Avg Living Sqft", 300,
                              int(clean_df.Sqft_living15.max()), 2000)
    lat          = st.number_input("Latitude",  value=47.5,  step=0.01, format="%.4f")
    lon          = st.number_input("Longitude", value=-122.2, step=0.01, format="%.4f")

# ─────────────────────────────────────────────────────────────────────────────
# Build prediction input row
# ─────────────────────────────────────────────────────────────────────────────
sale_year = 2015
pred_input = pd.DataFrame([{
    "Sqft_living":   sqft_living,
    "Sqft_lot":      sqft_lot,
    "Sqft_above":    sqft_above,
    "Sqft_basement": sqft_bsmt,
    "Grade":         grade,
    "Condition":     condition,
    "Bedrooms":      bedrooms,
    "Bathrooms":     bathrooms,
    "Floors":        floors,
    "View":          view,
    "Waterfront":    waterfront,
    "Yr_built":      yr_built,
    "Age":           sale_year - yr_built,
    "Renovated":     renovated,
    "Sqft_living15": sqft_l15,
    "Lat":           lat,
    "Long":          lon,
}])

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 1 — OVERVIEW & KPIs  ════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview & KPIs":
    st.markdown("# 🏠 House Price Intelligence Dashboard")
    st.caption("King County, WA House Sales Dataset · 21,613 Properties · 21 Features")

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    avg_p   = clean_df.Price.mean()
    med_p   = clean_df.Price.median()
    max_p   = clean_df.Price.max()
    min_p   = clean_df.Price.min()
    wf_pct  = clean_df.Waterfront.mean() * 100

    k1.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Average Price</div>
        <div class="kpi-value">${avg_p:,.0f}</div>
        <div class="kpi-sub">mean sale price</div></div>""", unsafe_allow_html=True)

    k2.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Median Price</div>
        <div class="kpi-value">${med_p:,.0f}</div>
        <div class="kpi-sub">50th percentile</div></div>""", unsafe_allow_html=True)

    k3.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Highest Sale</div>
        <div class="kpi-value">${max_p:,.0f}</div>
        <div class="kpi-sub">maximum price</div></div>""", unsafe_allow_html=True)

    k4.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Lowest Sale</div>
        <div class="kpi-value">${min_p:,.0f}</div>
        <div class="kpi-sub">minimum price</div></div>""", unsafe_allow_html=True)

    k5.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Waterfront %</div>
        <div class="kpi-value">{wf_pct:.1f}%</div>
        <div class="kpi-sub">of all properties</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Stats table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Statistical Summary — Key Features</div>', unsafe_allow_html=True)

    key_cols = ["Price","Sqft_living","Sqft_lot","Grade","Condition","Bedrooms","Bathrooms","Yr_built"]
    stats_df = clean_df[key_cols].agg(["mean","median","std","var","min","max"]).T
    stats_df.columns = ["Mean","Median","Std Dev","Variance","Min","Max"]
    stats_df["Mode"] = [clean_df[c].mode()[0] for c in key_cols]
    stats_df = stats_df[["Mean","Median","Mode","Std Dev","Variance","Min","Max"]]
    stats_df = stats_df.map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
    st.dataframe(stats_df, use_container_width=True)

    # ── Mini charts ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(clean_df, x="Price", nbins=60,
                           title="Sale Price Distribution",
                           color_discrete_sequence=["#5b6fd6"],
                           labels={"Price": "Sale Price ($)"})
        fig.update_layout(**CHART_THEME)
        fig.update_traces(marker_line_width=0.5, marker_line_color="#3d4270")
        fig.add_vline(x=avg_p, line_dash="dash", line_color="#6be3b0",
                      annotation_text=f"Mean ${avg_p:,.0f}", annotation_font_color="#6be3b0")
        show_chart(fig, "hist_price_kpi")

    with c2:
        grade_avg = clean_df.groupby("Grade")["Price"].mean().reset_index()
        fig = px.bar(grade_avg, x="Grade", y="Price",
                     title="Avg Price by Grade",
                     color="Price", color_continuous_scale="Viridis",
                     labels={"Price": "Avg Price ($)", "Grade": "Building Grade"})
        fig.update_layout(**CHART_THEME)
        show_chart(fig, "bar_grade_kpi")

    # ── Year Built trend ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Price Trend by Year Built</div>', unsafe_allow_html=True)
    yr_avg = clean_df.groupby("Yr_built")["Price"].mean().reset_index()
    fig = px.line(yr_avg, x="Yr_built", y="Price",
                  title="Average Sale Price by Year Built",
                  labels={"Yr_built": "Year Built", "Price": "Avg Price ($)"},
                  color_discrete_sequence=["#6be3b0"])
    fig.update_layout(**CHART_THEME)
    fig.update_traces(line_width=2)
    show_chart(fig, "line_year")

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 2 — DATA QUALITY  ═══════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧹 Data Quality":
    st.markdown("# 🧹 Data Quality Report")

    # ── Summary pills ─────────────────────────────────────────────────────────
    st.markdown("### Cleaning Summary")
    total_missing_before = int(meta["missing_before"].sum())
    pills = [
        ("Rows (before)",     f"{meta['before_rows']:,}",          "#5b6fd6"),
        ("Rows (after)",      f"{meta['after_rows']:,}",            "#6be3b0"),
        ("Duplicates rm.",    f"{meta['dups_removed']}",            "#e07b6b"),
        ("Missing (before)",  f"{total_missing_before:,} cells",   "#e07b6b"),
        ("Missing (after)",   "✓ 0 — All resolved",                 "#6be3b0"),
    ]
    pill_html = "".join(
        f'<span style="display:inline-block;background:#1e2130;border:1px solid {c};'
        f'border-radius:20px;padding:5px 16px;font-size:0.85rem;color:{c};margin:4px;">'
        f'<b>{l}</b>: {v}</span>'
        for l, v, c in pills
    )
    st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── Missing values BEFORE ─────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="section-header">Missing Values — Before Cleaning</div>', unsafe_allow_html=True)
        miss = meta["missing_before"][meta["missing_before"] > 0].sort_values(ascending=False)
        if miss.empty:
            st.success("✅ No missing values in the raw dataset.")
        else:
            miss_df = pd.DataFrame({
                "Column":  miss.index,
                "Missing": miss.values,
                "Pct (%)": (miss.values / meta["before_rows"] * 100).round(1),
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with c2:
        if PLOTLY_AVAILABLE:
            miss = meta["missing_before"][meta["missing_before"] > 0]
            if not miss.empty:
                miss_df = pd.DataFrame({
                    "Column":  miss.index,
                    "Pct (%)": (miss.values / meta["before_rows"] * 100).round(1),
                })
                fig = px.bar(miss_df, x="Pct (%)", y="Column", orientation="h",
                             title="Missing Value Columns (%)",
                             color="Pct (%)", color_continuous_scale="Reds")
                fig.update_layout(yaxis=dict(autorange="reversed"), **CHART_THEME)
                show_chart(fig, "miss_bar")
            else:
                st.info("No missing value chart — dataset is complete.")

    # ── After cleaning ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Missing Values — After Cleaning</div>', unsafe_allow_html=True)
    remaining = clean_df.isnull().sum().sum()
    if remaining == 0:
        st.success("✅ Zero missing values after cleaning.")
    else:
        st.warning(f"⚠️ {remaining} missing values remain.")

    # ── Outlier detection ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Outlier Detection (IQR Method)</div>', unsafe_allow_html=True)
    out_cols = ["Price","Sqft_living","Sqft_lot","Sqft_above","Sqft_basement","Grade"]
    out_rows = []
    for col in out_cols:
        q1, q3 = clean_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n_out = ((clean_df[col] < lo) | (clean_df[col] > hi)).sum()
        out_rows.append({"Column": col, "Q1": f"{q1:,.0f}", "Q3": f"{q3:,.0f}",
                         "IQR": f"{iqr:,.0f}", "Lower Fence": f"{lo:,.0f}",
                         "Upper Fence": f"{hi:,.0f}", "Outliers": n_out})
    st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    if PLOTLY_AVAILABLE:
        fig = px.box(clean_df, y="Price", x="Grade",
                     title="Price Distribution by Grade (Outliers Visible)",
                     labels={"Grade": "Building Grade", "Price": "Sale Price ($)"},
                     color="Grade")
        fig.update_layout(**CHART_THEME)
        show_chart(fig, "box_outlier")

    # ── Data types ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Data Types Before & After</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column": clean_df.columns,
        "Before": [str(meta["before_dtypes"].get(c, "new")) for c in clean_df.columns],
        "After":  [str(clean_df[c].dtype) for c in clean_df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ── Before vs After ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Before vs After Dataset Summary</div>', unsafe_allow_html=True)
    ba_df = pd.DataFrame({
        "Metric":  ["Total Rows", "Missing Values (cells)", "Duplicate Rows", "Columns"],
        "Before":  [
            meta["before_rows"],
            int(meta["missing_before"].sum()),
            meta["dups_removed"],
            len(raw_df.columns),
        ],
        "After":   [meta["after_rows"], 0, 0, len(clean_df.columns)],
    })
    st.dataframe(ba_df, use_container_width=True, hide_index=True)

    st.info(f"""
**Cleaning Impact Summary**
- **{meta['dups_removed']} duplicate rows** were detected and removed (rows went from {meta['before_rows']:,} → {meta['after_rows']:,}).
- **{int(meta['missing_before'].sum()):,} missing cells** were introduced across 5 key columns \
(`Price` ~3 %, `Sqft_living` ~4 %, `Grade` ~2 %, `Bathrooms` ~3 %, `Yr_built` ~2 %) \
and resolved via **median imputation**.
- **Invalid bedrooms** (e.g. 33 bedrooms — data entry errors) were filtered out.
- **Zero-price rows** were removed to prevent model distortion.
- **Engineered features** added: `Sale_Year`, `Sale_Month`, `Has_Basement`, `Renovated`, `Age`.
- `Age` values below 0 (house built after sale — data error) were clipped to 0.
""")

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 3 — VISUALIZATIONS  ═════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Visualizations":
    st.markdown("# 📈 Interactive Visualizations")
    st.caption("All charts are interactive — hover, zoom, and filter.")

    # ── 1. Price Histogram ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">1 · Sale Price Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(clean_df, x="Price", nbins=70, marginal="box",
                       title="Sale Price Distribution with Box Plot",
                       color_discrete_sequence=["#5b6fd6"],
                       labels={"Price": "Sale Price ($)"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "hist1")

    # ── 2. Bar chart — avg price by grade ────────────────────────────────────
    st.markdown('<div class="section-header">2 · Average Price by Building Grade</div>', unsafe_allow_html=True)
    grade_avg = clean_df.groupby("Grade")["Price"].mean().reset_index()
    fig = px.bar(grade_avg, x="Grade", y="Price",
                 title="Average Sale Price by Building Grade (1–13)",
                 color="Price", color_continuous_scale="Plasma",
                 labels={"Price": "Avg Price ($)", "Grade": "Building Grade"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "bar2")

    # ── 3. Line chart — monthly price trend ───────────────────────────────────
    st.markdown('<div class="section-header">3 · Monthly Price Trend</div>', unsafe_allow_html=True)
    trend = clean_df.copy()
    trend["YearMonth"] = trend["Sale_Year"].astype(str) + "-" + trend["Sale_Month"].astype(str).str.zfill(2)
    monthly = trend.groupby("YearMonth")["Price"].agg(["mean","median"]).reset_index().sort_values("YearMonth")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.YearMonth, y=monthly["mean"],
                             name="Mean Price", line=dict(color="#5b6fd6", width=2)))
    fig.add_trace(go.Scatter(x=monthly.YearMonth, y=monthly["median"],
                             name="Median Price", line=dict(color="#6be3b0", width=2, dash="dot")))
    fig.update_layout(title="Mean & Median Sale Price by Month",
                      xaxis_title="Month", yaxis_title="Price ($)",
                      xaxis_tickangle=-45, **CHART_THEME)
    show_chart(fig, "line3")

    # ── 4. Scatter — Sqft_living vs Price ────────────────────────────────────
    st.markdown('<div class="section-header">4 · Living Area vs Sale Price</div>', unsafe_allow_html=True)
    sample = clean_df.sample(min(3000, len(clean_df)), random_state=42)
    fig = px.scatter(sample, x="Sqft_living", y="Price",
                     color="Grade", size="Bathrooms",
                     hover_data=["Bedrooms","Yr_built","Condition"],
                     title="Living Area vs Sale Price (color=Grade, size=Bathrooms)",
                     color_continuous_scale="Viridis",
                     labels={"Sqft_living":"Living Area (sq ft)","Price":"Sale Price ($)"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "scatter4")

    # ── 5. Box plot — price vs bedrooms ──────────────────────────────────────
    st.markdown('<div class="section-header">5 · Price Distribution by Bedroom Count</div>', unsafe_allow_html=True)
    bed_df = clean_df[clean_df.Bedrooms.between(1,8)]
    fig = px.box(bed_df, x="Bedrooms", y="Price",
                 color="Bedrooms",
                 title="Sale Price Distribution by Number of Bedrooms",
                 labels={"Bedrooms":"Bedrooms","Price":"Sale Price ($)"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "box5")

    # ── 6. Heatmap — correlation ──────────────────────────────────────────────
    st.markdown('<div class="section-header">6 · Correlation Heatmap</div>', unsafe_allow_html=True)
    heat_cols = ["Price","Sqft_living","Sqft_lot","Sqft_above","Sqft_basement",
                 "Grade","Condition","Bedrooms","Bathrooms","Floors","View",
                 "Waterfront","Yr_built","Age","Sqft_living15"]
    corr = clean_df[heat_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Feature Correlation Matrix", **CHART_THEME, width=800, height=600)
    show_chart(fig, "heat6")

    # ── 7. Pairwise scatter matrix ────────────────────────────────────────────
    st.markdown('<div class="section-header">7 · Pairwise Relationships</div>', unsafe_allow_html=True)
    pair_cols = ["Price","Sqft_living","Grade","Bathrooms","Sqft_above"]
    sample2   = clean_df[pair_cols + ["Condition"]].sample(min(2000, len(clean_df)), random_state=1)
    fig = px.scatter_matrix(sample2,
                            dimensions=pair_cols,
                            color="Condition", color_continuous_scale="Viridis",
                            title="Pairwise Feature Relationships (sample 2000 rows)")
    fig.update_layout(**CHART_THEME)
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=3)
    show_chart(fig, "pair7")

    # ── 8. Violin — condition vs price ────────────────────────────────────────
    st.markdown('<div class="section-header">8 · Price Distribution by Condition (Violin)</div>', unsafe_allow_html=True)
    fig = px.violin(clean_df, x="Condition", y="Price",
                    box=True, points="outliers",
                    color="Condition",
                    title="Sale Price Violin Plot by Property Condition",
                    labels={"Condition":"Condition (1–5)","Price":"Sale Price ($)"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "violin8")

    # ── 9. Bar — waterfront premium ───────────────────────────────────────────
    st.markdown('<div class="section-header">9 · Waterfront vs Non-Waterfront Price</div>', unsafe_allow_html=True)
    wf_avg = (clean_df.groupby("Waterfront")["Price"].mean()
              .reset_index()
              .assign(Label=lambda d: d.Waterfront.map({0:"Non-Waterfront", 1:"Waterfront"})))
    fig = px.bar(wf_avg, x="Label", y="Price",
                 title="Average Sale Price: Waterfront vs Non-Waterfront",
                 color="Price", color_continuous_scale="Teal",
                 labels={"Price":"Avg Price ($)","Label":"Property Type"})
    fig.update_layout(**CHART_THEME)
    show_chart(fig, "bar_wf9")

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 4 — REGRESSION LAB  ═════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📐 Regression Lab":
    st.markdown("# 📐 Custom Regression Analysis")
    st.caption("Select features and explore a dynamic linear regression model.")

    reg_options = ["Sqft_living","Sqft_lot","Sqft_above","Sqft_basement",
                   "Grade","Condition","Bedrooms","Bathrooms","Floors",
                   "View","Yr_built","Age","Sqft_living15"]

    c1, c2 = st.columns([1, 2])
    with c1:
        sel_feats = st.multiselect(
            "Select 1–4 features for regression",
            reg_options, default=["Sqft_living", "Grade"]
        )
        if len(sel_feats) == 0:
            st.warning("Select at least one feature.")
            st.stop()
        x_axis = st.selectbox("X-axis for scatter", sel_feats, index=0)

    if sel_feats:
        from sklearn.linear_model import LinearRegression as LR
        from sklearn.metrics import r2_score as r2

        X_reg = clean_df[sel_feats].dropna()
        y_reg = clean_df.loc[X_reg.index, "Price"]
        model_reg = LR().fit(X_reg, y_reg)
        y_hat     = model_reg.predict(X_reg)
        r2_val    = r2(y_reg, y_hat)

        eq_parts = [f"{c:.2f}·{f}" for c, f in zip(model_reg.coef_, sel_feats)]
        equation = f"Price = {' + '.join(eq_parts)} + {model_reg.intercept_:.0f}"

        with c1:
            st.markdown("**Regression Equation:**")
            st.code(equation)
            st.metric("R² Score", f"{r2_val:.4f}")
            st.metric("Coefficients", len(model_reg.coef_))

        with c2:
            plot_df = X_reg.copy()
            plot_df["Price"]     = y_reg.values
            plot_df["Predicted"] = y_hat
            plot_df["Residual"]  = y_reg.values - y_hat

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df[x_axis], y=plot_df["Price"],
                mode="markers", name="Actual",
                marker=dict(color="#5b6fd6", size=4, opacity=0.5),
            ))
            fig.add_trace(go.Scatter(
                x=plot_df[x_axis], y=plot_df["Predicted"],
                mode="markers", name="Predicted",
                marker=dict(color="#6be3b0", size=3, opacity=0.4),
            ))
            sorted_df = plot_df.sort_values(x_axis)
            fig.add_trace(go.Scatter(
                x=sorted_df[x_axis], y=sorted_df["Predicted"],
                mode="lines", name="Trend",
                line=dict(color="#e07b6b", width=2, dash="dot"),
            ))
            fig.update_layout(
                title=f"Regression: {x_axis} vs Price  |  R² = {r2_val:.4f}",
                xaxis_title=x_axis, yaxis_title="Sale Price ($)", **CHART_THEME
            )
            show_chart(fig, "reg_scatter")

        st.markdown('<div class="section-header">Residual Analysis</div>', unsafe_allow_html=True)
        fig2 = px.scatter(plot_df, x="Predicted", y="Residual",
                          title="Residuals vs Predicted Values",
                          color="Residual", color_continuous_scale="RdBu",
                          labels={"Predicted":"Predicted Price ($)","Residual":"Residual ($)"})
        fig2.update_layout(**CHART_THEME)
        fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        show_chart(fig2, "residuals")

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 5 — ML MODELS  ══════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Models":
    st.markdown("# 🤖 Machine Learning Models")

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    # RMSE derived from the stored test-set MSE (same y_test / same split as R² and MAE)
    lr_eval = {**lr_metrics, "rmse": float(np.sqrt(lr_metrics["mse"]))}
    rf_eval = {**rf_metrics, "rmse": float(np.sqrt(rf_metrics["mse"]))}

    m1, m2, m3, m4 = st.columns(4)
    for col, label, key in [(m1,"R² Score","r2"),(m2,"MAE","mae"),(m3,"RMSE","rmse"),(m4,"MSE","mse")]:
        lr_v = lr_eval[key]
        rf_v = rf_eval[key]
        fmt  = "${:,.0f}" if key in ("mae","rmse","mse") else "{:.4f}"
        col.metric(f"LR — {label}",  fmt.format(lr_v))
        col.metric(f"RF — {label}",  fmt.format(rf_v),
                   delta=f"{((rf_v-lr_v)/abs(lr_v)*100):+.1f}%" if key=="r2" else None)

    eval_rows = [("R² (higher is better)", "r2",   "{:.4f}",   True),
                 ("MAE ($)",               "mae",  "${:,.0f}", False),
                 ("RMSE ($)",              "rmse", "${:,.0f}", False)]
    eval_df = pd.DataFrame({
        "Metric":            [r[0] for r in eval_rows],
        "Linear Regression": [r[2].format(lr_eval[r[1]]) for r in eval_rows],
        "Random Forest":     [r[2].format(rf_eval[r[1]]) for r in eval_rows],
        "Better Model":      [("Random Forest" if (rf_eval[r[1]] > lr_eval[r[1]]) == r[3] else "Linear Regression")
                              for r in eval_rows],
    })
    st.dataframe(eval_df, hide_index=True, use_container_width=True)
    st.caption(f"All metrics computed on the same held-out test set (n = {len(y_test):,}).")

    MODEL_COLORS = {"Linear Regression": "#5b6fd6", "Random Forest": "#6be3b0"}
    mc1, mc2 = st.columns(2)
    with mc1:
        r2_df = pd.DataFrame({"Model": list(MODEL_COLORS), "R²": [lr_eval["r2"], rf_eval["r2"]]})
        fig = px.bar(r2_df, x="Model", y="R²", color="Model", color_discrete_map=MODEL_COLORS,
                     title="R² Score (higher is better)", text=r2_df["R²"].map("{:.4f}".format))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(**CHART_THEME, showlegend=False, yaxis=dict(range=[0, 1], title="R²"))
        show_chart(fig, "metrics_r2")
    with mc2:
        err_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAE", "RMSE"],
            "Model":  ["Linear Regression"] * 2 + ["Random Forest"] * 2,
            "Value":  [lr_eval["mae"], lr_eval["rmse"], rf_eval["mae"], rf_eval["rmse"]],
        })
        fig = px.bar(err_df, x="Metric", y="Value", color="Model", barmode="group",
                     color_discrete_map=MODEL_COLORS, title="MAE vs RMSE (lower is better)",
                     text=err_df["Value"].map("${:,.0f}".format))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(**CHART_THEME, yaxis=dict(title="Error ($)", tickprefix="$", tickformat="~s"))
        show_chart(fig, "metrics_err")

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Actual vs Predicted Prices</div>', unsafe_allow_html=True)

    # Same test observations for both models (flattened to 1-D arrays to avoid index/shape issues)
    y_act = np.asarray(y_test, dtype=float).ravel()
    avp_models = {
        "Linear Regression": (np.asarray(y_pred_lr, dtype=float).ravel(), "#5b6fd6"),
        "Random Forest":     (np.asarray(y_pred_rf, dtype=float).ravel(), "#6be3b0"),
    }
    avp_view = st.radio("View", ["Side by side", "Linear Regression", "Random Forest", "Overlay both models"],
                        horizontal=True, key="avp_view")

    def avp_figure(labels, title):
        fig = go.Figure()
        for lb in labels:
            y_p, color = avp_models[lb]
            fig.add_trace(go.Scatter(x=y_act, y=y_p, mode="markers", name=lb,
                                     marker=dict(color=color, size=4, opacity=0.45 if len(labels) > 1 else 0.5),
                                     hovertemplate=f"{lb}<br>Actual: $%{{x:,.0f}}<br>Predicted: $%{{y:,.0f}}<extra></extra>"))
        lo = min(y_act.min(), *(avp_models[lb][0].min() for lb in labels))
        hi = max(y_act.max(), *(avp_models[lb][0].max() for lb in labels))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="white", dash="dash"), name="Perfect Fit",
                                 hoverinfo="skip"))
        fig.update_layout(title=title, xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)",
                          xaxis=dict(range=[lo, hi], tickprefix="$", tickformat="~s"),
                          yaxis=dict(range=[lo, hi], tickprefix="$", tickformat="~s"),
                          **CHART_THEME)
        return fig

    if avp_view == "Side by side":
        c1, c2 = st.columns(2)
        for col, label in [(c1, "Linear Regression"), (c2, "Random Forest")]:
            with col:
                show_chart(avp_figure([label], f"{label}: Actual vs Predicted"), f"avp_{label}")
    elif avp_view == "Overlay both models":
        show_chart(avp_figure(list(avp_models), "Actual vs Predicted: Linear Regression vs Random Forest"),
                   "avp_overlay")
        st.caption("Click a model name in the legend to hide or show it.")
    else:
        show_chart(avp_figure([avp_view], f"{avp_view}: Actual vs Predicted"), f"avp_single_{avp_view}")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Random Forest Feature Importance</div>', unsafe_allow_html=True)

    # Read directly from the already-trained Random Forest (no retraining)
    rf_importances = np.asarray(rf_pipeline.named_steps["model"].feature_importances_, dtype=float)
    try:
        rf_feat_names = [n.split("__", 1)[-1] for n in rf_pipeline.named_steps["prep"].get_feature_names_out()]
    except Exception:
        rf_feat_names = list(ALL_FEATURES)
    if len(rf_feat_names) != len(rf_importances):
        rf_feat_names = list(ALL_FEATURES)

    if len(rf_feat_names) == len(rf_importances):
        fi_full = (pd.DataFrame({"feature": rf_feat_names, "importance": rf_importances})
                   .sort_values("importance", ascending=False).reset_index(drop=True))
        fi_n = st.slider("Number of features to show", 5, len(fi_full), len(fi_full), key="fi_top_n")
        fi_show = fi_full.head(fi_n)
        fig = px.bar(fi_show, x="importance", y="feature", orientation="h",
                     title=f"Feature Importance — Random Forest (Top {fi_n})",
                     color="importance", color_continuous_scale="Viridis",
                     labels={"importance":"Importance Score","feature":"Feature"},
                     text=fi_show["importance"].map("{:.1%}".format))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>")
        fig.update_layout(**CHART_THEME, height=max(380, 30 * fi_n + 140))
        fig.update_layout(yaxis=dict(autorange="reversed", dtick=1),
                          xaxis=dict(range=[0, fi_show["importance"].max() * 1.15]))
        show_chart(fig, "fi")
    else:
        st.warning("Feature names do not match the trained Random Forest; showing stored importances.")
        fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
                     title="Feature Importance — Random Forest",
                     color="importance", color_continuous_scale="Viridis",
                     labels={"importance":"Importance Score","feature":"Feature"})
        fig.update_layout(**CHART_THEME)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        show_chart(fig, "fi")

    # ── Summary ───────────────────────────────────────────────────────────────
    # ── Key Findings text is generated from the trained RF's actual importances ──
    _fi = fi_full if len(rf_feat_names) == len(rf_importances) else importance_df
    _fi = _fi.sort_values("importance", ascending=False).reset_index(drop=True)
    _why = {
        "Sqft_living":   "interior living area is the main driver of a home's size and value.",
        "Grade":         "construction and finish quality separates price tiers.",
        "Lat":           "north-south position captures neighbourhood and market location.",
        "Long":          "east-west position captures neighbourhood and market location.",
        "Sqft_living15": "the size of nearby homes reflects the local market level.",
    }
    _why_default = "contributes to the model's price splits."
    findings = "\n".join(
        f"{i}. **{r.feature}** ({r.importance:.1%}) — {_why.get(r.feature, _why_default)}"
        for i, r in enumerate(_fi.head(5).itertuples(), 1)
    )
    _loc = _fi.set_index("feature")["importance"].reindex(["Lat", "Long"]).sum()
    findings += (f"\n\nLat and Long together account for {_loc:.1%} of importance. "
                 "Importance reflects how much the model relies on a feature, not a causal effect.")

    st.markdown("---")
    st.markdown("""
### 📝 Key Findings & Conclusions

**Most Influential Features** (from Random Forest importance):
{findings}

**Model Performance:**
| Model | R² | MAE |
|---|---|---|
| Linear Regression | {:.3f} | ${:,.0f} |
| Random Forest | {:.3f} | ${:,.0f} |

Random Forest outperforms Linear Regression by {:.1f} percentage points of R² on the same test set.

**Observed Inconsistencies:**
- One record had 33 bedrooms (likely a data entry error) — removed.
- `Yr_renovated = 0` indicates no renovation — converted to a binary `Renovated` flag.
- `Age` was derived as `Sale_Year - Yr_built`; negative values clipped to 0.

**Impact of Cleaning:**
- No rows lost from imputation (dataset had zero missing values).
- Feature engineering added 5 new informative columns.
- Outlier bedrooms removed to prevent model distortion.
""".format(
        lr_metrics["r2"], lr_metrics["mae"],
        rf_metrics["r2"], rf_metrics["mae"],
        (rf_metrics["r2"] - lr_metrics["r2"]) * 100,
        findings=findings,
    ))

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  PAGE 6 — PREDICT  ════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Price":
    st.markdown("# 🔮 House Price Predictor")
    st.caption("Adjust inputs in the sidebar and click Predict.")

    model_choice = st.radio("Choose Model", ["Random Forest (recommended)", "Linear Regression"],
                             horizontal=True)

    if st.button("🏠 Predict Price", type="primary"):
        pipeline = rf_pipeline if "Random" in model_choice else lr_pipeline
        price    = pipeline.predict(pred_input)[0]

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e2130,#252840);
    border:2px solid #5b6fd6;border-radius:16px;padding:32px;text-align:center;margin:20px 0;">
  <div style="color:#8b93c4;font-size:0.85rem;text-transform:uppercase;letter-spacing:2px;">
      Estimated Sale Price</div>
  <div style="color:#6be3b0;font-size:3.5rem;font-weight:800;margin:12px 0;">
      ${price:,.0f}</div>
  <div style="color:#8b93c4;font-size:0.82rem;">
      Model: {model_choice}</div>
</div>""", unsafe_allow_html=True)

        pct = (clean_df.Price < price).mean() * 100
        st.metric("Percentile in Dataset", f"{pct:.0f}th",
                  help="What percentage of sold homes cost less than this estimate")

        st.markdown("### Input Summary")
        st.dataframe(pred_input.T.rename(columns={0: "Value"}), use_container_width=True)

        if PLOTLY_AVAILABLE:
            mn, mx = clean_df.Price.min(), clean_df.Price.max()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=price,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Estimated Price ($)", "font":{"color":"#c5cef0"}},
                number={"prefix":"$","font":{"color":"#6be3b0","size":40}},
                gauge={
                    "axis":    {"range":[mn, mx], "tickcolor":"#8b93c4"},
                    "bar":     {"color":"#6be3b0"},
                    "bgcolor": "#1a1d2e",
                    "steps":   [
                        {"range":[mn, clean_df.Price.quantile(.33)],  "color":"#2a1d3e"},
                        {"range":[clean_df.Price.quantile(.33), clean_df.Price.quantile(.66)], "color":"#1e2d3e"},
                        {"range":[clean_df.Price.quantile(.66), mx],  "color":"#1e3e2a"},
                    ],
                    "threshold":{"line":{"color":"#5b6fd6","width":3},
                                 "thickness":0.75,"value":clean_df.Price.median()},
                }
            ))
            fig.update_layout(**CHART_THEME, height=350)
            show_chart(fig, "gauge")
            st.caption("Blue threshold line = dataset median price")
