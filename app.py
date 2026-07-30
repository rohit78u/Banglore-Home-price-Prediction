"""Bengaluru property intelligence dashboard."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "notebooks" / "banglore_home_prices_model.pickle"
COLUMNS_PATH = APP_DIR / "notebooks" / "columns.json"
DATA_PATH = APP_DIR / "data" / "Bengaluru_House_Data.csv"

st.set_page_config(
    page_title="NEST — Bengaluru Property Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
          :root { --ink:#edf4f2; --muted:#9eb5ae; --mint:#91f5c5; --dark:#071a1a; --panel:rgba(17,43,42,.74); }
          .stApp { background: radial-gradient(circle at 82% -5%, #1a5a4c 0, transparent 29%), radial-gradient(circle at -10% 65%, #123b3d 0, transparent 32%), #071a1a; color:var(--ink); font-family:Manrope,sans-serif; }
          #MainMenu, footer, header {visibility:hidden;}
          .block-container { max-width: 1220px; padding: 2rem 2.2rem 4rem; }
          .hero { position:relative; overflow:hidden; padding:2.6rem 2.8rem; border:1px solid rgba(164,255,216,.19); border-radius:28px; background:linear-gradient(120deg,rgba(20,61,57,.96),rgba(10,31,31,.87)); box-shadow:0 28px 55px rgba(0,0,0,.28), inset 0 1px rgba(255,255,255,.08); }
          .hero:after { content:''; position:absolute; width:300px; height:300px; right:-80px; bottom:-160px; border-radius:50%; background:radial-gradient(circle,#8df4c333,transparent 67%); }
          .eyebrow { color:var(--mint); font:500 .72rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.16em; margin-bottom:.8rem; }
          .hero h1 { margin:0; max-width:730px; color:#f4fbf8; font-size:clamp(2.15rem,5vw,4.2rem); font-weight:800; line-height:1.05; letter-spacing:-.065em; }
          .hero p { max-width:610px; margin:.9rem 0 0; color:#c3d4cf; font-size:1rem; line-height:1.7; }
          .stButton > button { border:0; border-radius:12px; background:linear-gradient(135deg,#a2f8ca,#6ae4ba); color:#07251f; font-weight:800; min-height:3.1rem; box-shadow:0 8px 20px #47d99a24; transition:transform .2s ease, box-shadow .2s ease; }
          .stButton > button:hover { transform:translateY(-2px); box-shadow:0 13px 25px #47d99a38; color:#07251f; }
          .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb='select'] > div { border-radius:12px !important; border-color:rgba(168,238,209,.22)!important; background:#0c2928!important; color:#eef7f2!important; }
          div[data-testid='stMetric'] { padding:1.05rem 1.15rem; border:1px solid rgba(159,236,205,.14); border-radius:17px; background:linear-gradient(145deg,rgba(34,70,66,.86),rgba(12,37,36,.88)); box-shadow:10px 12px 22px rgba(0,0,0,.15), inset 1px 1px rgba(255,255,255,.04); }
          div[data-testid='stMetricLabel'] { color:#a8beb7; font-size:.77rem; text-transform:uppercase; letter-spacing:.08em; }
          div[data-testid='stMetricValue'] { color:#f1faf6; font-weight:800; }
          .result { margin-top:1.1rem; padding:1.4rem 1.5rem; border-radius:18px; background:linear-gradient(135deg,#163e38,#0d302e); border:1px solid #78dcb13d; box-shadow:0 16px 30px rgba(0,0,0,.2), inset 0 1px #ffffff12; }
          .result-label { color:#9cb8af; font:500 .69rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }.result-price { color:#a6f7cc; font-size:2.35rem; font-weight:800; letter-spacing:-.06em; }.section-title { margin:2.3rem 0 .25rem; color:#f0f9f5; font-size:1.35rem; font-weight:800; letter-spacing:-.04em; }.section-copy { color:#9ab0aa; margin:0 0 1.1rem; }.micro { color:#8ba59d; font-size:.76rem; }
          div[data-testid='stTabs'] button { color:#a6bbb5; font-weight:700; } div[data-testid='stTabs'] button[aria-selected='true'] { color:#9cf3c5; }
          @media (max-width: 700px) { .block-container{padding:1rem 1rem 3rem}.hero{padding:2rem 1.5rem}.hero h1{font-size:2.35rem} }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_assets() -> tuple[object, list[str]]:
    """Load and validate the trained estimator and its expected feature order."""
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Expected files in notebooks/.")
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with COLUMNS_PATH.open(encoding="utf-8") as columns_file:
        features = json.load(columns_file).get("data_columns", [])
    if len(features) < 4 or features[:3] != ["total_sqft", "bath", "bhk"]:
        raise ValueError("The model feature schema is invalid or unsupported.")
    return model, features


def parse_sqft(value: object) -> float | None:
    """Convert raw floor-area text, including ranges, into a usable square-foot value."""
    if pd.isna(value):
        return None
    text = str(value).replace(",", "").strip().lower()
    if "-" in text:
        parts = text.split("-", maxsplit=1)
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return None
    numeric = "".join(character for character in text if character.isdigit() or character == ".")
    try:
        return float(numeric) if numeric else None
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    """Load a compact, clean data frame used by the insight visualizations."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("Market data file is missing from data/.")
    raw = pd.read_csv(DATA_PATH)
    required = {"location", "size", "total_sqft", "bath", "price"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
    data = raw.copy()
    data["bhk"] = pd.to_numeric(data["size"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data["total_sqft"] = data["total_sqft"].map(parse_sqft)
    data["location"] = data["location"].astype(str).str.strip()
    data = data.dropna(subset=["location", "total_sqft", "bath", "bhk", "price"])
    data = data[(data["total_sqft"] >= 250) & (data["price"] > 0) & (data["price"] < 1000)]
    data["price_per_sqft"] = data["price"] * 100_000 / data["total_sqft"]
    return data


def predict_price(model: object, features: list[str], location: str, sqft: int, bath: int, bhk: int) -> float:
    values = np.zeros(len(features), dtype=float)
    values[:3] = [sqft, bath, bhk]
    location_index = {name: index for index, name in enumerate(features)}.get(location.lower())
    if location_index is not None:
        values[location_index] = 1
    prediction = float(model.predict([values])[0])
    return max(prediction, 0.0)


def format_inr_lakhs(value: float) -> str:
    return f"₹ {value:,.1f} L"


def render_estimator(model: object, features: list[str], market: pd.DataFrame) -> None:
    st.markdown("<div class='eyebrow'>AI valuation workspace</div><h2 class='section-title' style='margin-top:0'>Estimate your home’s value</h2><p class='section-copy'>Enter the key property details to receive a practical market estimate for your chosen neighbourhood.</p>", unsafe_allow_html=True)
    form_col, insight_col = st.columns([1.1, 0.9], gap="large")
    locations = sorted(features[3:])
    with form_col:
        with st.form("valuation_form", border=False):
            location = st.selectbox("Neighbourhood", locations, index=locations.index("whitefield") if "whitefield" in locations else 0)
            input_left, input_right = st.columns(2)
            with input_left:
                sqft = st.number_input("Home area (sq ft)", min_value=300, max_value=20_000, value=1200, step=50)
                bhk = st.slider("Bedrooms", min_value=1, max_value=8, value=2)
            with input_right:
                bath = st.slider("Bathrooms", min_value=1, max_value=8, value=2)
            st.caption("All prices are shown in Indian rupees (lakhs).")
            submitted = st.form_submit_button("Estimate home value", use_container_width=True)
        if submitted:
            estimate = predict_price(model, features, location, sqft, bath, bhk)
            st.session_state["estimate"] = {"value": estimate, "location": location, "sqft": sqft, "bhk": bhk, "bath": bath}
        if estimate := st.session_state.get("estimate"):
            st.markdown(f"<div class='result'><div class='result-label'>Estimated market value</div><div class='result-price'>{format_inr_lakhs(estimate['value'])}</div><div class='micro'>{estimate['bhk']} BHK · {estimate['bath']} bath · {estimate['sqft']:,} sq ft in {estimate['location'].title()}</div></div>", unsafe_allow_html=True)
    with insight_col:
        active_location = st.session_state.get("estimate", {}).get("location", "whitefield")
        location_data = market[market["location"].str.lower() == active_location]
        reference_data = location_data if not location_data.empty else market
        st.metric(f"Typical price in {active_location.title()}", format_inr_lakhs(reference_data["price"].median()))
        st.metric("Typical price per sq ft", f"₹ {reference_data['price_per_sqft'].median():,.0f}")
        st.metric("Typical home size", f"{reference_data['total_sqft'].median():,.0f} sq ft")
        st.caption("Valuations are indicative estimates based on historic listing data, not a formal appraisal.")


def render_insights(market: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Understand the market</div><p class='section-copy'>Use these citywide benchmarks to compare prices, neighbourhoods, and home sizes.</p>", unsafe_allow_html=True)
    overview, neighbourhoods, property_mix = st.tabs(["Citywide prices", "Compare neighbourhoods", "Size and price"])
    chart_layout = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"color": "#c8d9d3", "family": "Manrope"}, "margin": {"l": 8, "r": 8, "t": 44, "b": 8}}
    with overview:
        metrics = st.columns(4)
        common_bhk = int(market["bhk"].mode().iat[0])
        metrics[0].metric("Typical Bengaluru home price", format_inr_lakhs(market["price"].median()))
        metrics[1].metric("Typical home size", f"{market['total_sqft'].median():,.0f} sq ft")
        metrics[2].metric("Typical price per sq ft", f"₹ {market['price_per_sqft'].median():,.0f}")
        metrics[3].metric("Most common home layout", f"{common_bhk} BHK")
        fig = px.histogram(market, x="price", nbins=45, histnorm="percent", title="Where Bengaluru home prices usually fall", color_discrete_sequence=["#91f5c5"])
        fig.update_layout(**chart_layout, xaxis_title="Home price (lakhs)", yaxis_title="Share of homes (%)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with neighbourhoods:
        comparison = st.radio("Compare neighbourhoods by", ["Typical home price", "Typical price per sq ft"], horizontal=True)
        location_summary = market.groupby("location", as_index=False).agg(typical_price=("price", "median"), typical_price_per_sqft=("price_per_sqft", "median"), properties=("price", "size")).query("properties >= 8")
        if comparison == "Typical home price":
            metric_column, chart_title, axis_title = "typical_price", "Neighbourhoods with the highest typical home prices", "Typical home price (lakhs)"
        else:
            metric_column, chart_title, axis_title = "typical_price_per_sqft", "Neighbourhoods with the highest typical price per sq ft", "Typical price per sq ft (₹)"
        location_summary = location_summary.nlargest(12, metric_column).sort_values(metric_column)
        fig = px.bar(location_summary, x=metric_column, y="location", orientation="h", color=metric_column, color_continuous_scale=["#1e4d47", "#91f5c5"], title=chart_title)
        fig.update_layout(**chart_layout, coloraxis_showscale=False, xaxis_title=axis_title, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with property_mix:
        selected_location = st.selectbox("Choose a neighbourhood", sorted(market["location"].unique()), key="market_location")
        subset = market[market["location"] == selected_location]
        fig = px.scatter(subset, x="total_sqft", y="price", color="bhk", size="bath", hover_data={"bhk": True, "bath": True, "total_sqft": ":,.0f", "price": ":.1f"}, title=f"How home size relates to price in {selected_location.title()}", color_continuous_scale="Mint", labels={"bhk": "Bedrooms", "bath": "Bathrooms", "total_sqft": "Home size (sq ft)", "price": "Home price (lakhs)"})
        fig.update_layout(**chart_layout, xaxis_title="Home size (sq ft)", yaxis_title="Home price (lakhs)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    inject_styles()
    try:
        model, features = load_model_assets()
        market = load_market_data()
    except (FileNotFoundError, ValueError, OSError, pickle.UnpicklingError) as error:
        st.error("The application could not load its required data assets.")
        st.caption(str(error))
        st.stop()

    st.markdown("<section class='hero'><div class='eyebrow'>Bengaluru · residential intelligence</div><h1>Make your next property move with clarity.</h1><p>NEST combines a trained valuation model with local market signals to turn property details into a confident, fast estimate.</p></section>", unsafe_allow_html=True)
    st.markdown("<p class='micro' style='margin:1rem 0 2rem'>MODEL-POWERED ESTIMATES · HISTORIC LISTING ANALYSIS · BUILT FOR DISCOVERY</p>", unsafe_allow_html=True)
    render_estimator(model, features, market)
    render_insights(market)
    st.markdown("<p class='micro' style='margin-top:3rem'>NEST / BENGALURU PROPERTY INTELLIGENCE · Estimates are informational and should be verified before a transaction.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
