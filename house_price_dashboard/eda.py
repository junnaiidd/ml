"""
eda.py
------
Standalone Exploratory Data Analysis script for King County House Sales dataset.
Prints summaries and saves HTML Plotly charts to /eda_output/.
Run: python eda.py
"""

import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
OUT_DIR   = os.path.join(BASE_DIR, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print("\n--- Missing values ---")
miss = df.isnull().sum()
print(miss[miss > 0] if miss.sum() > 0 else "No missing values.")

print("\n--- Numeric summary ---")
print(df.describe().T[["mean","std","min","25%","50%","75%","max"]].to_string())

print("\n--- Duplicates ---")
print(f"  {df.duplicated().sum()} duplicate rows")

print("\n--- Top correlations with Price ---")
corr = df.select_dtypes("number").corr()["Price"].sort_values(ascending=False)
print(corr.head(15))

print("\n--- Key categorical-like columns ---")
print("Waterfront:", df["Waterfront"].value_counts().to_dict())
print("View:",       df["View"].value_counts().to_dict())
print("Condition:",  df["Condition"].value_counts().to_dict())
print("Grade range:", df["Grade"].min(), "–", df["Grade"].max())

try:
    import plotly.express as px
    fig = px.histogram(df, x="Price", nbins=60, title="Price Distribution — King County")
    fig.write_html(os.path.join(OUT_DIR, "price_distribution.html"))
    print(f"\n✅ EDA chart saved → {OUT_DIR}/price_distribution.html")
except ImportError:
    print("Plotly not available — install it to generate charts.")

print("\nEDA complete.")
