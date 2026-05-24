"""
Ethiopian House Price Predictor
================================
Compares Linear Regression, Random Forest, and Gradient Boosting
on a realistic Ethiopian real estate dataset (Addis Ababa market).

Author : Estifanos Abera
Dataset: 500 houses across 19 Addis Ababa neighborhoods (2025-2026 prices in ETB)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────

def load_data(path="ethiopian_house_prices.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()

    # Encode categorical columns
    le_zone = LabelEncoder()
    le_condition = LabelEncoder()
    le_neighborhood = LabelEncoder()

    df["zone_enc"]         = le_zone.fit_transform(df["zone"])
    df["condition_enc"]    = le_condition.fit_transform(df["condition"])
    df["neighborhood_enc"] = le_neighborhood.fit_transform(df["neighborhood"])

    feature_cols = [
        "area_sqm", "bedrooms", "bathrooms", "age_years", "floors",
        "has_parking", "has_garden", "near_cbd", "near_ring_road",
        "near_school", "near_hospital",
        "zone_enc", "condition_enc", "neighborhood_enc"
    ]

    X = df[feature_cols]
    y = df["price_etb"]

    return X, y, feature_cols, le_neighborhood, le_zone, le_condition

# ─────────────────────────────────────────────
# 2. Model Definitions
# ─────────────────────────────────────────────

def build_models():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
    }

# ─────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

        results[name] = {
            "model":    model,
            "MAE":      mae,
            "RMSE":     rmse,
            "R2":       r2,
            "CV_R2_mean": cv.mean(),
            "CV_R2_std":  cv.std(),
        }

    return results

def print_results(results):
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — Ethiopian House Price Prediction (ETB)")
    print("=" * 70)
    header = f"{'Model':<22} {'MAE (ETB)':>15} {'RMSE (ETB)':>15} {'R²':>7} {'CV R²':>10}"
    print(header)
    print("-" * 70)

    for name, r in results.items():
        print(f"{name:<22} {r['MAE']:>15,.0f} {r['RMSE']:>15,.0f} "
              f"{r['R2']:>7.4f} {r['CV_R2_mean']:>7.4f}±{r['CV_R2_std']:.3f}")

    best = max(results, key=lambda k: results[k]["R2"])
    print("-" * 70)
    print(f"\n  Best model: {best}  (R² = {results[best]['R2']:.4f})")
    print("=" * 70)

# ─────────────────────────────────────────────
# 4. Predict a new house
# ─────────────────────────────────────────────

NEIGHBORHOODS = [
    "Akaki Kality","Ayat","Bole","Bole Bulbula","CMC","Gerji","Goro",
    "Jemo","Kazanchis","Kirkos","Kolfe","Lebu","Lideta","Megenagna",
    "Old Airport","Piassa","Sarbet","Summit","Yeka"
]

ZONE_MAP = {
    "Bole":"premium","Old Airport":"premium","Kazanchis":"premium","Sarbet":"premium",
    "Megenagna":"mid-high","CMC":"mid-high","Gerji":"mid-high","Summit":"mid-high",
    "Yeka":"mid","Lebu":"mid","Ayat":"mid","Bole Bulbula":"mid",
    "Goro":"emerging","Akaki Kality":"emerging","Kolfe":"emerging","Jemo":"emerging",
    "Lideta":"mid","Piassa":"mid","Kirkos":"mid"
}

CBD_AREAS    = {"Bole","Old Airport","Kazanchis","Sarbet","Lideta","Piassa","Kirkos"}
RING_AREAS   = {"Bole","Old Airport","Megenagna","CMC","Summit","Bole Bulbula"}
SCHOOL_AREAS = {"Bole","Old Airport","Kazanchis","Sarbet","Megenagna","CMC","Gerji",
                "Yeka","Lideta","Piassa","Kirkos","Ayat"}
HOSP_AREAS   = {"Bole","Old Airport","Kazanchis","Megenagna","Lideta","Kirkos"}

def predict_new_house(best_model, le_nb, le_zone, le_cond, feature_cols):
    print("\n" + "=" * 50)
    print("  PREDICT A NEW HOUSE PRICE")
    print("=" * 50)
    print("Available neighborhoods:")
    for i, nb in enumerate(NEIGHBORHOODS, 1):
        print(f"  {i:2}. {nb}")

    idx = int(input("\nEnter neighborhood number: ")) - 1
    neighborhood = NEIGHBORHOODS[idx]
    zone         = ZONE_MAP[neighborhood]

    area      = float(input("Enter area (sqm): "))
    bedrooms  = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    age       = int(input("Enter house age (years): "))
    floors    = int(input("Enter number of floors: "))
    parking   = int(input("Has parking? (1=Yes, 0=No): "))
    garden    = int(input("Has garden? (1=Yes, 0=No): "))
    cond      = input("Condition (new/good/fair/poor): ").strip().lower()

    row = pd.DataFrame([{
        "area_sqm":       area,
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "age_years":      age,
        "floors":         floors,
        "has_parking":    parking,
        "has_garden":     garden,
        "near_cbd":       int(neighborhood in CBD_AREAS),
        "near_ring_road": int(neighborhood in RING_AREAS),
        "near_school":    int(neighborhood in SCHOOL_AREAS),
        "near_hospital":  int(neighborhood in HOSP_AREAS),
        "zone_enc":       le_zone.transform([zone])[0],
        "condition_enc":  le_cond.transform([cond])[0],
        "neighborhood_enc": le_nb.transform([neighborhood])[0],
    }])[feature_cols]

    price = best_model.predict(row)[0]
    usd   = price / 155  # approx exchange rate 2025

    print(f"\n  Neighborhood : {neighborhood} ({zone})")
    print(f"  Area         : {area:.0f} sqm | {bedrooms}BR/{bathrooms}BA | Age: {age}yr")
    print(f"  Condition    : {cond.capitalize()}")
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Predicted Price: {price:>15,.0f} ETB  ║")
    print(f"  ║  (~${usd:>12,.0f} USD)              ║")
    print(f"  ╚══════════════════════════════════════╝\n")

# ─────────────────────────────────────────────
# 5. Feature Importance (for tree models)
# ─────────────────────────────────────────────

def print_feature_importance(model, feature_cols, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        m = model.named_steps.get("model")
        if hasattr(m, "coef_"):
            importances = np.abs(m.coef_)
        else:
            return
    else:
        return

    pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print(f"\n  Top Features — {model_name}:")
    for feat, imp in pairs[:8]:
        bar = "█" * int(imp * 40 / pairs[0][1])
        print(f"  {feat:<22} {bar} {imp:.4f}")

# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    print("\n  Loading Ethiopian house price dataset...")
    df = load_data()
    print(f"  Dataset: {len(df)} houses across {df['neighborhood'].nunique()} neighborhoods")
    print(f"  Price range: {df['price_etb'].min():,.0f} – {df['price_etb'].max():,.0f} ETB")
    print(f"  Avg price  : {df['price_etb'].mean():,.0f} ETB")

    X, y, feature_cols, le_nb, le_zone, le_cond = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n  Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    print("  Training 4 models...")

    models  = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    print_results(results)

    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]
    print_feature_importance(best_model, feature_cols, best_name)

    predict_new_house(best_model, le_nb, le_zone, le_cond, feature_cols)

if __name__ == "__main__":
    main()
