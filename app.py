"""
Streamlit Web App — Ethiopian House Price Predictor
====================================================
Run with: streamlit run app.py
"""

import streamlit as st
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

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ethiopian House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .metric-box {
    background: linear-gradient(135deg, #1a6b3c, #2ecc71);
    border-radius: 12px; padding: 1rem 1.4rem; color: white;
    margin-bottom: 0.5rem;
  }
  .metric-box h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
  .metric-box p  { margin: 0; font-size: 1.5rem; font-weight: 700; }
  .price-card {
    background: linear-gradient(135deg, #0d3b1e, #1a6b3c);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
  }
  .price-card .price { font-size: 2.5rem; font-weight: 800; color: #2ecc71; }
  .price-card .usd   { font-size: 1.1rem; opacity: 0.8; margin-top: 0.3rem; }
  .badge {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  }
  .badge-premium  { background: #d4af37; color: #000; }
  .badge-mid-high { background: #2ecc71; color: #000; }
  .badge-mid      { background: #3498db; color: #fff; }
  .badge-emerging { background: #95a5a6; color: #000; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

ZONE_MAP = {
    "Bole":"premium","Old Airport":"premium","Kazanchis":"premium","Sarbet":"premium",
    "Megenagna":"mid-high","CMC":"mid-high","Gerji":"mid-high","Summit":"mid-high",
    "Yeka":"mid","Lebu":"mid","Ayat":"mid","Bole Bulbula":"mid",
    "Goro":"emerging","Akaki Kality":"emerging","Kolfe":"emerging","Jemo":"emerging",
    "Lideta":"mid","Piassa":"mid","Kirkos":"mid",
}
CBD_AREAS    = {"Bole","Old Airport","Kazanchis","Sarbet","Lideta","Piassa","Kirkos"}
RING_AREAS   = {"Bole","Old Airport","Megenagna","CMC","Summit","Bole Bulbula"}
SCHOOL_AREAS = {"Bole","Old Airport","Kazanchis","Sarbet","Megenagna","CMC","Gerji",
                "Yeka","Lideta","Piassa","Kirkos","Ayat"}
HOSP_AREAS   = {"Bole","Old Airport","Kazanchis","Megenagna","Lideta","Kirkos"}

NEIGHBORHOODS = sorted(ZONE_MAP.keys())

FEATURE_COLS = [
    "area_sqm","bedrooms","bathrooms","age_years","floors",
    "has_parking","has_garden","near_cbd","near_ring_road",
    "near_school","near_hospital","zone_enc","condition_enc","neighborhood_enc"
]

# ── Data & Model (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_and_train():
    df = pd.read_csv("ethiopian_house_prices.csv")

    le_nb   = LabelEncoder().fit(df["neighborhood"])
    le_zone = LabelEncoder().fit(df["zone"])
    le_cond = LabelEncoder().fit(df["condition"])

    df["zone_enc"]         = le_zone.transform(df["zone"])
    df["condition_enc"]    = le_cond.transform(df["condition"])
    df["neighborhood_enc"] = le_nb.transform(df["neighborhood"])

    X = df[FEATURE_COLS]
    y = df["price_etb"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        "Ridge Regression":  Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    }

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        cv   = cross_val_score(m, X_train, y_train, cv=kf, scoring="r2")
        results[name] = {
            "model": m,
            "MAE":   mean_absolute_error(y_test, pred),
            "RMSE":  np.sqrt(mean_squared_error(y_test, pred)),
            "R2":    r2_score(y_test, pred),
            "CV":    cv.mean(),
            "CV_std": cv.std(),
        }

    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]

    return df, results, best_model, best_name, le_nb, le_zone, le_cond

# ── App ────────────────────────────────────────────────────────────────────────

st.title("🏠 Ethiopian House Price Predictor")
st.caption("Addis Ababa real estate — ML-powered price estimates in ETB · 2025–2026 market data")

df, results, best_model, best_name, le_nb, le_zone, le_cond = load_and_train()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "📁 Dataset"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter house details")
    col1, col2, col3 = st.columns(3)

    with col1:
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)
        zone = ZONE_MAP[neighborhood]
        badge_cls = f"badge-{zone.replace('-','-')}"
        st.markdown(f"<span class='badge {badge_cls}'>{zone.upper()}</span>", unsafe_allow_html=True)
        area      = st.number_input("Area (sqm)", 30, 1000, 120)
        bedrooms  = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1, 6, 2)

    with col2:
        age       = st.slider("Age of house (years)", 0, 40, 5)
        floors    = st.slider("Number of floors", 1, 5, 2)
        condition = st.selectbox("Condition", ["new", "good", "fair", "poor"])
        st.write("")

    with col3:
        has_parking = st.checkbox("Has Parking", value=True)
        has_garden  = st.checkbox("Has Garden")
        st.write("")
        st.markdown("**Auto-detected from neighborhood:**")
        st.write(f"Near CBD: {'✅' if neighborhood in CBD_AREAS else '❌'}")
        st.write(f"Near Ring Road: {'✅' if neighborhood in RING_AREAS else '❌'}")
        st.write(f"Near School: {'✅' if neighborhood in SCHOOL_AREAS else '❌'}")
        st.write(f"Near Hospital: {'✅' if neighborhood in HOSP_AREAS else '❌'}")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "area_sqm":       area,
            "bedrooms":       bedrooms,
            "bathrooms":      bathrooms,
            "age_years":      age,
            "floors":         floors,
            "has_parking":    int(has_parking),
            "has_garden":     int(has_garden),
            "near_cbd":       int(neighborhood in CBD_AREAS),
            "near_ring_road": int(neighborhood in RING_AREAS),
            "near_school":    int(neighborhood in SCHOOL_AREAS),
            "near_hospital":  int(neighborhood in HOSP_AREAS),
            "zone_enc":       le_zone.transform([zone])[0],
            "condition_enc":  le_cond.transform([condition])[0],
            "neighborhood_enc": le_nb.transform([neighborhood])[0],
        }])[FEATURE_COLS]

        price = best_model.predict(row)[0]
        usd   = price / 155

        st.markdown(f"""
        <div class='price-card'>
          <p style='margin:0;font-size:1rem;opacity:0.7;'>Predicted Price — {neighborhood}</p>
          <div class='price'>{price:,.0f} ETB</div>
          <div class='usd'>≈ ${usd:,.0f} USD</div>
          <p style='margin-top:1rem;font-size:0.8rem;opacity:0.6;'>
            Model: {best_name} · {area}sqm · {bedrooms}BR/{bathrooms}BA · {condition.capitalize()}
          </p>
        </div>
        """, unsafe_allow_html=True)

        per_sqm = price / area
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Price per sqm", f"{per_sqm:,.0f} ETB")
        c2.metric("Neighborhood zone", zone.capitalize())
        c3.metric("Model used", best_name)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Comparison
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performance Comparison")
    st.caption("All models trained on 80% of 500 Ethiopian house listings, evaluated on 20% holdout + 5-fold CV")

    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "MAE (ETB)":  f"{r['MAE']:,.0f}",
            "RMSE (ETB)": f"{r['RMSE']:,.0f}",
            "R² Score":   f"{r['R2']:.4f}",
            "CV R² (mean±std)": f"{r['CV']:.4f} ± {r['CV_std']:.3f}",
            "Best?": "⭐" if name == best_name else ""
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("R² Scores")
    chart_data = pd.DataFrame({
        "Model": list(results.keys()),
        "R²":    [r["R2"] for r in results.values()]
    }).set_index("Model")
    st.bar_chart(chart_data)

    st.write("")
    st.subheader("Feature Importance")
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": imp})
        fi_df = fi_df.sort_values("Importance", ascending=True).set_index("Feature")
        st.bar_chart(fi_df)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Dataset
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Dataset — {len(df)} Ethiopian houses")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total houses", len(df))
    c2.metric("Neighborhoods", df["neighborhood"].nunique())
    c3.metric("Avg price", f"{df['price_etb'].mean():,.0f} ETB")
    c4.metric("Price range", f"{df['price_etb'].min()/1e6:.1f}M – {df['price_etb'].max()/1e6:.0f}M")

    st.write("")
    st.subheader("Average Price by Neighborhood (ETB)")
    avg_by_nb = df.groupby("neighborhood")["price_etb"].mean().sort_values(ascending=True)
    st.bar_chart(avg_by_nb)

    st.write("")
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)
