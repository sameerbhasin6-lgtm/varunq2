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
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)
        return -(rev_bundle + rev_indiv)

    bounds = []
    for i in range(n_prods):
        max_w = np.max(wtp_matrix[:, i])
        bounds.append((0, max_w * 1.5)) 
    bounds.append((0, np.max(bundle_sum_values)))

    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_breakdown(df, products, optimal_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    bundle_price = optimal_prices[n_prods]
    
    rows = []
    for i in range(len(df)):
        s_indiv = np.sum(np.maximum(wtp_matrix[i] - indiv_prices, 0))
        s_bundle = bundle_sum_values[i] - bundle_price
        
        decision = "None"
        revenue = 0
        surplus = 0
        items = "-"
        
        if s_bundle >= s_indiv and s_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            surplus = s_bundle
            items = "All Items"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_indices = np.where(wtp_matrix[i] >= indiv_prices)[0]
            items = ", ".join([products[k] for k in bought_indices])
            revenue = np.sum(indiv_prices[bought_indices])
            
        rows.append({
            "Customer ID": i + 1,
            "Decision": decision,
            "Items Bought": items.replace("Samsung_", "").replace("_", " "),
            "Revenue": revenue,
            "Consumer Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_curve(df, products, optimal_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    max_val = np.max(bundle_sum_values)
    price_points = np.linspace(0, max_val, 100)
    demand = []
    for bp in price_points:
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bp
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        demand.append(np.sum(buy_bundle))
    return pd.DataFrame({"Price": price_points, "Demand": demand})

# --- MAIN APP ---

def main():
    with st.sidebar:
        st.markdown("### Control Center")
        st.info("System Status: **Online**")
        st.markdown("**About**: This engine uses Differential Evolution to maximize revenue via Mixed Bundling.")
        st.write("---")
        st.caption("v1.2.0 • PriceAi")

    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Dynamic Pricing Optimizer</h1>
        <div class="header-subtitle">AI-Driven Mixed Bundling Strategy</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    products = df.columns.tolist()
    
    with st.spinner("🧠 AI Solver running..."):
        baseline_rev = calculate_baseline(df, products)
        opt_prices, max_rev = solve_pricing(df, products)
        customer_df = get_customer_breakdown(df, products, opt_prices)
        
        total_surplus = customer_df['Consumer Surplus'].sum()
        uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
        bundle_price = opt_prices[-1]
        sum_indiv_opt = np.sum(opt_prices[:-1])
        discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
        bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100
        
        # --- SECTION 1: KPIS ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Projected Revenue</div>
                <div class="kpi-value">₹{max_rev:,.0f}</div>
                <div class="kpi-delta text-green">▲ {uplift:.1f}% vs Baseline</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Consumer Surplus</div>
                <div class="kpi-value">₹{total_surplus:,.0f}</div>
                <div class="kpi-delta text-blue">Value Retained</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Bundle Adoption</div>
                <div class="kpi-value">{bundle_adoption:.0f}%</div>
                <div class="kpi-delta text-indigo">Conversion Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Bundle Discount</div>
                <div class="kpi-value">{discount:.1f}%</div>
                <div class="kpi-delta" style="color:#64748b;">Effective Savings</div>
            </div>
            """, unsafe_allow_html=True)

        # --- SECTION 2: PRICING STRATEGY (FIXED) ---
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.subheader("Recommended Pricing Mix")
        
        # FIX: We construct the string without indentation to prevent code-block rendering
        cards_html = ""
        
        # Individual Cards
        for i, prod in enumerate(products):
            p_opt = opt_prices[i]
            clean_name = prod.replace("Samsung_", "").replace("_", " ")
            cards_html += f"""
            <div class="price-card">
                <div class="price-prod-name">{clean_name}</div>
                <div class="price-amount">₹{p_opt:,.0f}</div>
            </div>
            """
        
        # Bundle Card
        cards_html += f"""
        <div class="price-card featured">
            <div class="price-prod-name">⚡ All-in Bundle</div>
            <div class="price-amount">₹{bundle_price:,.0f}</div>
        </div>
        """
        
        # Final Container HTML - Removing newlines to ensure clean markdown parsing
        final_html = f'<div class="pricing-container">{cards_html}</div>'.replace('\n', '')
        
        st.markdown(final_html, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # --- SECTION 3: INSIGHTS ---
        c_left, c_right = st.columns([1, 2], gap="large")
        
        with c_left:
            st.subheader("Strategic Insights")
            
            if discount > 15:
                strategy_title = "Volume Maximization"
                strategy_desc = "Deep discounting allows for higher volume penetration, offsetting lower margins."
                color_border = "#10b981"
            else:
                strategy_title = "Premium Extraction"
                strategy_desc = "High willingness-to-pay suggests keeping prices high to maximize margin per unit."
                color_border = "#6366f1"

            st.markdown(f"""
            <div class="insight-box" style="border-left-color: {color_border}">
                <div class="insight-title">🎯 {strategy_title}</div>
                <div class="insight-body">{strategy_desc}</div>
            </div>
            <div class="insight-box" style="border-left-color: #f59e0b">
                <div class="insight-title">📢 Marketing Angle</div>
                <div class="insight-body">
                    Highlight the <b>₹{(sum_indiv_opt - bundle_price):,.0f}</b> savings prominently.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.subheader("Consumer Choice Simulation")
            st.dataframe(
                customer_df,
                column_config={
                    "Customer ID": st.column_config.NumberColumn(format="#%d"),
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Consumer Surplus": st.column_config.ProgressColumn(format="₹%d", max_value=int(customer_df['Consumer Surplus'].max())),
                    "Decision": st.column_config.TextColumn(),
                    "Items Bought": st.column_config.TextColumn(),
                },
                use_container_width=True,
                height=400,
                hide_index=True
            )

        # --- SECTION 4: CHART ---
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Bundle Demand Sensitivity")
        
        demand_data = generate_demand_curve(df, products, opt_prices)
        
        fig = px.area(demand_data, x="Price", y="Demand", labels={"Price": "Bundle Price (₹)", "Demand": "Buyers"})
        fig.add_vline(x=bundle_price, line_dash="dash", line_color="#10b981", annotation_text="Optimal Price")
        fig.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#64748b"),
            hovermode="x unified"
        )
        fig.update_traces(line_color='#6366f1', fillcolor='rgba(99, 102, 241, 0.1)')
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
