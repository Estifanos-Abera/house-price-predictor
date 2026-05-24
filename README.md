# 🏠 Ethiopian House Price Predictor

> A machine learning project that predicts residential property prices across **19 Addis Ababa neighborhoods**, built on realistic 2025–2026 Ethiopian market data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn) ![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 About

This project started as a beginner ML exercise and has grown into a full Ethiopian real estate prediction system. It compares **4 machine learning models** on a custom dataset of **500 houses** across Addis Ababa, with prices and features grounded in real market research.

**Key improvements over v1:**
- Dataset expanded from 6 → 500 houses
- 3 features → 14 features (location proximity, condition, parking, etc.)
- 1 model → 4 models compared with cross-validation
- Console-only → full interactive web app (Streamlit)
- Generic data → realistic Ethiopian/Addis Ababa market prices in ETB

---

## 🏙️ Neighborhoods Covered

| Zone | Neighborhoods | Price Range |
|------|--------------|-------------|
| **Premium** | Bole, Old Airport, Kazanchis, Sarbet | 40M – 150M ETB |
| **Mid-High** | Megenagna, CMC, Gerji, Summit | 20M – 60M ETB |
| **Mid** | Yeka, Ayat, Lebu, Lideta, Piassa, Kirkos, Bole Bulbula | 8M – 35M ETB |
| **Emerging** | Goro, Akaki Kality, Kolfe, Jemo | 4M – 20M ETB |

> Prices reflect 2025–2026 market data. Bole & Kazanchis command premiums due to proximity to embassies, the airport, and international businesses.

---

## 🧠 Models Compared

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Linear Regression** | Fast, interpretable | Assumes linearity |
| **Ridge Regression** | Handles multicollinearity | Still linear |
| **Random Forest** | Captures non-linear patterns, robust | Less interpretable |
| **Gradient Boosting** | Highest accuracy on tabular data | Slower to train |

---

## 🗂️ Project Structure

```
house-price-predictor/
├── ethiopian_house_prices.csv   ← 500-row dataset (14 features)
├── house_price_predictor.py     ← CLI model training & comparison
├── app.py                       ← Streamlit web app
├── README.md                    ← This file
└── requirements.txt             ← Dependencies
```

---

## ⚙️ Features in the Dataset

| Feature | Description |
|---------|-------------|
| `neighborhood` | One of 19 Addis Ababa areas |
| `zone` | premium / mid-high / mid / emerging |
| `area_sqm` | House size in square meters |
| `bedrooms` / `bathrooms` | Room counts |
| `age_years` | How old the house is |
| `floors` | Number of storeys |
| `has_parking` | 1 = yes, 0 = no |
| `has_garden` | 1 = yes, 0 = no |
| `near_cbd` | Close to city center |
| `near_ring_road` | Access to Addis ring road |
| `near_school` / `near_hospital` | Proximity to amenities |
| `condition` | new / good / fair / poor |
| `price_etb` | Target: price in Ethiopian Birr |

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Estifanos-Abera/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run the CLI version
```bash
python house_price_predictor.py
```

### 3b. Run the Web App
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 📊 Sample Results

```
══════════════════════════════════════════════════════════════════════
  MODEL COMPARISON — Ethiopian House Price Prediction (ETB)
══════════════════════════════════════════════════════════════════════
Model                      MAE (ETB)      RMSE (ETB)      R²      CV R²
──────────────────────────────────────────────────────────────────────
Linear Regression          3,200,000       4,800,000  0.8821  0.877±0.021
Ridge Regression           3,150,000       4,750,000  0.8840  0.880±0.019
Random Forest              1,900,000       2,900,000  0.9612  0.955±0.011
Gradient Boosting          1,750,000       2,700,000  0.9680  0.962±0.009
```

**Best model: Gradient Boosting** — captures non-linear neighborhood effects and condition interactions better than linear models.

---

## 📰 Research Paper

See [`research_paper.md`](research_paper.md) for the accompanying article:
*"Comparing Machine Learning Approaches for Ethiopian Real Estate Price Prediction"*

---

## 🇪🇹 Market Context

Ethiopian real estate — especially in Addis Ababa — has seen **8–10% annual price growth** forecasted through 2030, driven by diaspora investment and urban expansion. This project reflects that market reality:

- **Bole** remains the most expensive area ($1,500–$2,000/sqm)
- **CMC and Ayat** offer emerging value ($800–$1,300/sqm)
- Proximity to the **ring road, CBD, and embassies** significantly raises prices
- **Condition** and **age** are strong predictors after location

---

## 👤 Author

**Estifanos Abera** — [GitHub](https://github.com/Estifanos-Abera)

Built to learn ML, grown to reflect real Ethiopian real estate dynamics.
