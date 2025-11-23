import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PriceAi | Dynamic Pricing Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM AESTHETICS & CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Header Styling */
    .header-container {
        padding: 1rem 0rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    /* Metric Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .text-green { color: #10b981; }
    .text-blue { color: #3b82f6; }
    .text-indigo { color: #6366f1; }

    /* Pricing Cards Layout */
    .pricing-container {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 20px;
        margin-bottom: 40px;
        justify-content: center; /* Center cards if few */
    }
    .price-card {
        flex: 1;
        min-width: 200px;
        max-width: 300px;
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s;
    }
    .price-card:hover {
        transform: translateY(-5px);
    }
    .price-card.featured {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        border: none;
        color: white;
        transform: scale(1.05);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.3);
        z-index: 10;
    }
    .price-prod-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .price-card.featured .price-prod-name { color: #e0e7ff; }
    .price-amount {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    .price-card.featured .price-amount { color: white; }

    /* Insights Section */
    .insight-box {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .insight-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .insight-body {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.5;
    }
    
    .divider {
        height: 1px;
        background-color: #e2e8f0;
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    FILE_NAME = "Varun.csv"
    try:
        df = pd.read_csv(FILE_NAME)
        return df
    except FileNotFoundError:
        st.error(f"🔴 System Error: Data file '{FILE_NAME}' not found.")
        st.stop()
        
# --- 2. OPTIMIZATION ENGINE ---

def calculate_baseline(df, products):
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def solve_pricing(df, products):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        rev_bundle = np.sum(buy_bundle) * bundle_price
