# 🏠 House Price Intelligence Dashboard — King County, WA

A production-style, interactive machine learning dashboard for predicting and analyzing house prices.  
Built with **Python · Pandas · Scikit-learn · Plotly · Streamlit**.

---

## 📁 Project Structure

```
house_price_dashboard/
├── app/
│   └── app.py              ← Streamlit dashboard (main UI)
├── data/
│   └── house_prices.csv    ← King County House Sales (21,613 rows, 21 features)
├── models/
│   └── model.pkl           ← Trained model bundle (auto-generated)
├── eda.py                  ← Standalone EDA script
├── model.py                ← Model training script
├── requirements.txt
└── README.md
```

---

## 📦 Dataset — King County House Sales

| Column | Description |
|--------|-------------|
| Price | Sale price (target) |
| Bedrooms / Bathrooms | Room counts |
| Sqft_living | Interior living area (sq ft) |
| Sqft_lot | Lot size (sq ft) |
| Floors | Number of floors |
| Waterfront | Waterfront property flag (0/1) |
| View | View quality index (0–4) |
| Condition | Property condition (1–5) |
| Grade | Building quality grade (1–13) |
| Sqft_above / Sqft_basement | Above & below-grade sq ft |
| Yr_built / Yr_renovated | Year built and last renovated |
| zipcode / Lat / Long | Location data |
| Sqft_living15 | Avg living sqft of 15 nearest neighbors |

---

## ⚙️ Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (run once)
python model.py

# 3. Launch the dashboard
streamlit run app/app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview & KPIs** | Key metrics, stats table, price trend by year built |
| **Data Quality** | Missing values, outlier detection, cleaning report |
| **Visualizations** | 9 interactive Plotly charts |
| **Regression Lab** | Dynamic custom regression with any feature combination |
| **ML Models** | LR vs RF comparison, feature importance, actual vs predicted |
| **Predict Price** | User-input price predictor with gauge chart |

---

## 🤖 Model Performance

| Model | R² | MAE |
|-------|-----|-----|
| Linear Regression | ~0.695 | ~$124,000 |
| Random Forest | ~0.877 | ~$68,700 |

---

## 📌 Key Findings

- **Lat/Long** — Geographic location is the most important signal
- **Sqft_living** — Living area is the top structural predictor
- **Grade** — Building quality grade is highly correlated with price (~0.667)
- **Waterfront** — Waterfront homes sell for ~3× more on average
- Random Forest outperforms Linear Regression by ~18% in R²
