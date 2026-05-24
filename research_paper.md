# Comparing Machine Learning Approaches for Ethiopian Real Estate Price Prediction

**Author:** Estifanos Abera  
**Date:** May 2026  
**Keywords:** machine learning, real estate, Ethiopia, Addis Ababa, regression, Random Forest, Gradient Boosting

---

## Abstract

Accurate house price prediction is a critical problem in real estate markets, yet most published work focuses on Western or Asian cities with mature, data-rich markets. This paper presents a machine learning study applied to the **Addis Ababa residential property market** — one of Africa's fastest-growing real estate environments. We build a synthetic-but-realistic dataset of 500 properties across 19 neighborhoods, grounded in 2025–2026 market research, and compare four regression models: Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting. Results show that tree-based ensemble methods significantly outperform linear approaches (R² of 0.97 vs. 0.88), with neighborhood zone and area emerging as the dominant price drivers. This work provides a foundation for data-driven property valuation tools in the Ethiopian context.

---

## 1. Introduction

The Ethiopian real estate market, particularly in Addis Ababa, has experienced rapid growth in recent years. Prime neighborhoods such as Bole and Kazanchis command prices between $1,500 and $2,000 per square meter, while emerging areas like Ayat and Akaki Kality offer entry points at $800–$1,300 per square meter [1]. Despite this dynamism, formal property valuation in Ethiopia remains largely manual and subjective, making data-driven tools both novel and valuable.

House price prediction is a well-studied problem in machine learning. Studies consistently show that location features dominate price variation, followed by structural attributes like size, age, and condition [2]. However, the unique characteristics of African urban real estate — including informal markets, proximity effects tied to specific local landmarks (embassies, ring roads, hospitals), and the influence of diaspora investment — make direct transfer of Western models unreliable.

This study makes three contributions:
1. A publicly available Ethiopian house price dataset reflecting real Addis Ababa neighborhood price hierarchies.
2. A systematic comparison of four ML regression algorithms on this dataset.
3. An analysis of feature importance specific to the Ethiopian housing context.

---

## 2. Dataset & Features

### 2.1 Data Collection & Construction

Due to the limited availability of open Ethiopian real estate datasets, we constructed a synthetic dataset using prices derived from published market reports including the Miles Consulting Residential Report (H2 2023, inflation-adjusted to 2026) and real estate listings from Ethiopian Property Centre [3]. The dataset contains **500 residential properties** across **19 Addis Ababa neighborhoods**.

### 2.2 Features

We include 14 input features across three categories:

**Structural features:** area (sqm), bedrooms, bathrooms, age (years), floors, condition (new/good/fair/poor), has_parking, has_garden.

**Location features:** neighborhood (19 categories), zone (premium/mid-high/mid/emerging).

**Proximity features (binary):** near_cbd, near_ring_road, near_school, near_hospital. These were assigned based on established geographic knowledge of Addis Ababa infrastructure.

**Target variable:** price in Ethiopian Birr (ETB).

### 2.3 Neighborhood Price Hierarchy

| Zone | Representative Areas | Price Range (ETB/sqm) |
|------|---------------------|----------------------|
| Premium | Bole, Old Airport, Kazanchis | 220,000 – 300,000 |
| Mid-High | CMC, Megenagna, Summit | 150,000 – 210,000 |
| Mid | Yeka, Ayat, Lideta, Piassa | 100,000 – 160,000 |
| Emerging | Akaki Kality, Jemo, Kolfe | 80,000 – 120,000 |

---

## 3. Methods

### 3.1 Data Preprocessing

Categorical variables (neighborhood, zone, condition) were encoded using scikit-learn's `LabelEncoder`. A train/test split of 80%/20% was applied with a fixed random seed for reproducibility. Linear models received StandardScaler normalization; tree-based models did not require it.

### 3.2 Models

**Linear Regression (OLS)** serves as our baseline, assuming a linear additive relationship between features and price.

**Ridge Regression** extends OLS with L2 regularization (α = 10), addressing multicollinearity between correlated spatial features (e.g., near_cbd and neighborhood).

**Random Forest** uses an ensemble of 200 decision trees (max_depth=12), capturing non-linear interactions between features — for instance, the combined effect of location and condition.

**Gradient Boosting** uses sequential tree building (300 estimators, learning_rate=0.05, max_depth=5) to iteratively minimize prediction error. This approach typically achieves the highest accuracy on structured tabular data [4].

### 3.3 Evaluation

Each model was evaluated using:
- **MAE (Mean Absolute Error):** average absolute price prediction error in ETB
- **RMSE (Root Mean Squared Error):** penalizes large errors more heavily
- **R² Score:** proportion of price variance explained
- **5-Fold Cross-Validation R²:** tests generalization beyond a single train/test split

---

## 4. Results

### 4.1 Model Performance

| Model | MAE (ETB) | RMSE (ETB) | R² | CV R² |
|-------|-----------|------------|-----|-------|
| Linear Regression | ~3.2M | ~4.8M | 0.882 | 0.877 ± 0.021 |
| Ridge Regression | ~3.1M | ~4.7M | 0.884 | 0.880 ± 0.019 |
| Random Forest | ~1.9M | ~2.9M | 0.961 | 0.955 ± 0.011 |
| Gradient Boosting | ~1.7M | ~2.7M | 0.968 | 0.962 ± 0.009 |

Gradient Boosting achieves the best results across all metrics. The gap between linear and tree-based models is substantial — an R² improvement from 0.88 to 0.97 — indicating that house prices in Addis Ababa exhibit significant **non-linear patterns** that linear models cannot capture.

### 4.2 Feature Importance

Analysis of the Random Forest and Gradient Boosting models consistently identifies these top features:

1. **neighborhood_enc** — The most powerful predictor. Bole properties can be 5× more expensive than Jemo at the same size.
2. **area_sqm** — Strong linear relationship with price; larger homes command proportionally higher prices.
3. **zone_enc** — Neighborhood zone (premium/mid-high/mid/emerging) captures broad location tiers.
4. **condition_enc** — A "new" property commands ~15% premium over "good" condition.
5. **age_years** — Depreciation of roughly 1.2% per year observed in the data.
6. **near_cbd** — Proximity to the city center adds ~8% to price.
7. **near_ring_road** — Access to the ring road adds ~5%.
8. **has_garden / has_parking** — Secondary but consistent premiums of 4–6%.

### 4.3 Linear vs. Non-Linear Models

The underperformance of linear models is explained by several interactions:
- **Condition × Age:** an old house in poor condition loses far more value than a linear model predicts.
- **Neighborhood × Size:** a 400sqm house in Bole is disproportionately more valuable than in Kolfe — the premium is multiplicative, not additive.
- **Proximity thresholds:** being "near the CBD" matters most in mid-tier neighborhoods; premium neighborhoods already price this in.

---

## 5. Discussion

### 5.1 Ethiopian Market Specifics

Several factors make Ethiopian real estate unique and affect model design:

**Diaspora investment** drives demand in premium areas. Bole and Kazanchis attract buyers from the Ethiopian diaspora in the US, Europe, and Gulf states, creating demand that is relatively price-inelastic and disconnected from local income levels [1].

**Infrastructure as a proxy for development.** Ring road proximity is a strong predictor not just of commute convenience but of overall neighborhood development quality — paved roads, consistent electricity, water supply.

**Condition matters more at the extremes.** "New" construction commands a larger premium in Ethiopia than in mature markets, because the supply of quality new housing is limited and buyers are willing to pay significantly to avoid renovation costs.

### 5.2 Limitations

- The dataset is synthetic-realistic, not sourced from live transactions. Actual listing data from platforms like Ethiopia Property Centre would improve accuracy.
- The model does not account for **plot ownership type** (lease vs. freehold) which is legally and economically significant in Ethiopia.
- **Inflation** is not modeled dynamically; the dataset reflects a static 2025–2026 snapshot.

### 5.3 Future Work

- Collect real transaction data to validate and retrain models.
- Add features: building type (villa/apartment/condominium), lease term remaining, distance to specific landmarks (Bole Airport, Meskel Square, universities).
- Explore geospatial models using latitude/longitude coordinates.
- Build a price trend forecaster using time-series data.

---

## 6. Conclusion

This study demonstrates that **Gradient Boosting is the most effective model** for Ethiopian house price prediction, outperforming linear regression by a margin of 8.6% in R² score and cutting average prediction error by nearly half. The dominant predictors are location-based (neighborhood, zone, CBD proximity), consistent with real estate research globally, but with Ethiopia-specific weightings that reflect the unique dynamics of Addis Ababa's rapidly developing market.

The accompanying open-source tool — including the dataset, CLI predictor, and Streamlit web app — provides a practical foundation for anyone studying or working in Ethiopian real estate.

---

## References

[1] The Africanvestor. *"Housing Prices in Addis Ababa (2026)."* theafricanvestor.com, April 2026.

[2] Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.* 3rd ed. O'Reilly, 2022.

[3] Miles Consulting. *Residential Real Estate Market Report — Addis Ababa H2 2023.*

[4] Chen, T. & Guestrin, C. *"XGBoost: A Scalable Tree Boosting System."* KDD 2016.

[5] Bamboo Routes. *"Average Property Price in Addis Ababa (2025)."* bambooroutes.com, Aug 2025.

---

*This paper was written as part of the house-price-predictor open-source project by Estifanos Abera. Dataset and code available at: github.com/Estifanos-Abera/house-price-predictor*
