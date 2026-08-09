"""
model.py
--------
Trains Linear Regression and Random Forest models on the King County House Sales dataset.
Saves the trained pipeline (preprocessing + model) to models/model.pkl.
Run this once before launching the Streamlit app.
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load & Engineer Features
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "house_prices.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

df = pd.read_csv(DATA_PATH)

# Parse year/month from Date column
df["Sale_Year"]  = df["Date"].str[:4].astype(int)
df["Sale_Month"] = df["Date"].str[4:6].astype(int)

# Derived features
df["Has_Basement"] = (df["Sqft_basement"] > 0).astype(int)
df["Renovated"]    = (df["Yr_renovated"] > 0).astype(int)
df["Age"]          = df["Sale_Year"] - df["Yr_built"]

# ─────────────────────────────────────────────
# 2. Feature Selection
# ─────────────────────────────────────────────
# Dataset is fully numeric — no categorical OHE needed
NUM_FEATURES = [
    "Sqft_living",    # Interior living space sq ft
    "Sqft_lot",       # Lot size sq ft
    "Sqft_above",     # Sq ft above basement
    "Sqft_basement",  # Sq ft of basement
    "Grade",          # Overall grade (1-13)
    "Condition",      # Overall condition (1-5)
    "Bedrooms",       # Number of bedrooms
    "Bathrooms",      # Number of bathrooms
    "Floors",         # Number of floors
    "View",           # View quality index (0-4)
    "Waterfront",     # Waterfront property flag (0/1)
    "Yr_built",       # Year built
    "Age",            # Age at time of sale
    "Renovated",      # Whether ever renovated
    "Sqft_living15",  # Avg living sqft of 15 nearest neighbors
    "Lat",            # Latitude (location proxy)
    "Long",           # Longitude (location proxy)
]

CAT_FEATURES = []
TARGET = "Price"

df_model = df[NUM_FEATURES + [TARGET]].copy()

# ─────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, NUM_FEATURES),
])

# ─────────────────────────────────────────────
# 4. Train / Test Split
# ─────────────────────────────────────────────
X = df_model[NUM_FEATURES]
y = df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 5. Linear Regression
# ─────────────────────────────────────────────
lr_pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", LinearRegression()),
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

lr_metrics = {
    "r2":  r2_score(y_test, y_pred_lr),
    "mae": mean_absolute_error(y_test, y_pred_lr),
    "mse": mean_squared_error(y_test, y_pred_lr),
}
print("=== Linear Regression ===")
print(f"  R²  : {lr_metrics['r2']:.4f}")
print(f"  MAE : ${lr_metrics['mae']:,.0f}")
print(f"  MSE : {lr_metrics['mse']:,.0f}")

# ─────────────────────────────────────────────
# 6. Random Forest
# ─────────────────────────────────────────────
rf_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
])

rf_pipeline = Pipeline([
    ("prep",  rf_preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rf_metrics = {
    "r2":  r2_score(y_test, y_pred_rf),
    "mae": mean_absolute_error(y_test, y_pred_rf),
    "mse": mean_squared_error(y_test, y_pred_rf),
}
print("\n=== Random Forest ===")
print(f"  R²  : {rf_metrics['r2']:.4f}")
print(f"  MAE : ${rf_metrics['mae']:,.0f}")
print(f"  MSE : {rf_metrics['mse']:,.0f}")

# ─────────────────────────────────────────────
# 7. Feature Importance (RF)
# ─────────────────────────────────────────────
importances = rf_pipeline.named_steps["model"].feature_importances_

importance_df = (
    pd.DataFrame({"feature": NUM_FEATURES, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

# ─────────────────────────────────────────────
# 8. Save bundle
# ─────────────────────────────────────────────
bundle = {
    "lr_pipeline":    lr_pipeline,
    "rf_pipeline":    rf_pipeline,
    "lr_metrics":     lr_metrics,
    "rf_metrics":     rf_metrics,
    "num_features":   NUM_FEATURES,
    "cat_features":   CAT_FEATURES,
    "all_features":   NUM_FEATURES,
    "importance_df":  importance_df,
    "X_test":         X_test,
    "y_test":         y_test,
    "y_pred_lr":      y_pred_lr,
    "y_pred_rf":      y_pred_rf,
}
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(bundle, MODEL_PATH)
print(f"\n✅  Model bundle saved → {MODEL_PATH}")
