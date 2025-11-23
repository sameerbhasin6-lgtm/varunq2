import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PriceAi | Strategic Pricing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM AESTHETICS & CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
        background-color: #f8fafc;
    }

    /* Header */
    .header-container {
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .header-title { font-size: 2rem; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -0.02em; }
    .header-subtitle { font-size: 1rem; color: #64748b; margin-top: 0.2rem; }

    /* KPI Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        flex: 1;
    }
    .kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #1e293b; margin: 0.25rem 0; }
    .kpi-delta { font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 4px; }
    .text-green { color: #10b981; }
    .text-indigo { color: #6366f1; }
    .text-slate { color: #64748b; }

    /* Pricing Hero Section */
    .pricing-wrapper {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    .price-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        min-width: 180px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        position: relative;
    }
    .price-card-title { font-size: 0.85rem; font-weight: 600; color: #64748b; margin-bottom: 0.5rem; text-transform: uppercase; }
    .price-card-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    
    /* The Bundle Card - Hero Design */
    .bundle-card {
        background: radial-gradient(circle at top right, #4f46e5, #3b82f6);
        color: white;
        transform: scale(1.05);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.4);
        border: none;
        z-index: 2;
    }
    .bundle-card .price-card-title { color: rgba(255,255,255,0.8); }
    .bundle-card .price-card-value { color: white; font-size: 2rem; }
    .badge {
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        background: #f59e0b;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Strategy Box */
    .strategy-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .strategy-header { font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;}
    
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
            "Customer ID": i + 1, "Decision": decision, 
            "Items Bought": items.replace("Samsung_", "").replace("_", " "),
            "Revenue": revenue, "Consumer Surplus": surplus
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
    # Title Section
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">PriceAi Strategy Engine</h1>
        <div class="header-subtitle">Optimizing Mixed-Bundling Revenue & Market Segmentation</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    products = df.columns.tolist()
    
    with st.spinner("🔄 Synthesizing Market Data & Optimizing Prices..."):
        baseline_rev = calculate_baseline(df, products)
        opt_prices, max_rev = solve_pricing(df, products)
        customer_df = get_customer_breakdown(df, products, opt_prices)
        
        # Core Metrics
        total_surplus = customer_df['Consumer Surplus'].sum()
        uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
        bundle_price = opt_prices[-1]
        sum_indiv_opt = np.sum(opt_prices[:-1])
        discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
        bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100
        
        # --- 1. KPI ROW ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Optimized Revenue</div><div class="kpi-value">₹{max_rev:,.0f}</div><div class="kpi-delta text-green">▲ {uplift:.1f}% vs Baseline</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Bundle Conversion</div><div class="kpi-value">{bundle_adoption:.0f}%</div><div class="kpi-delta text-indigo">of Total Market</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Bundle Discount</div><div class="kpi-value">{discount:.1f}%</div><div class="kpi-delta text-slate">Effective Savings</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Consumer Surplus</div><div class="kpi-value">₹{total_surplus:,.0f}</div><div class="kpi-delta text-slate">Value Retained</div></div>""", unsafe_allow_html=True)

        # --- 2. PRICING HERO SECTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏷️ Optimal Pricing Configuration")
        
        # Build HTML for Pricing
        price_html = ""
        # Individual
        for i, prod in enumerate(products):
            p_opt = opt_prices[i]
            clean_name = prod.replace("Samsung_", "").replace("_", " ")
            price_html += f"""
            <div class="price-card">
                <div class="price-card-title">{clean_name}</div>
                <div class="price-card-value">₹{p_opt:,.0f}</div>
            </div>
            """
        # Bundle (Hero)
        price_html += f"""
        <div class="price-card bundle-card">
            <div class="badge">Recommended</div>
            <div class="price-card-title">All-In Bundle</div>
            <div class="price-card-value">₹{bundle_price:,.0f}</div>
        </div>
        """
        
        st.markdown(f'<div class="pricing-wrapper">{price_html}</div>', unsafe_allow_html=True)

        # --- 3. TABBED ANALYSIS ---
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📢 Marketing Strategy", "📊 Customer Segments", "📈 Demand Simulation"])
        
        # TAB 1: MARKETING INSIGHTS
        with tab1:
            m_col1, m_col2 = st.columns([1.5, 1])
            
            with m_col1:
                st.markdown("#### 🧠 Strategic Rationale")
                
                # Dynamic Logic for Text
                if discount > 15:
                    anchor_text = "The high individual prices serve as **strong anchors**, making the bundle discount feel massive. This 'Decoy Effect' pushes indecisive buyers toward the full package."
                    segment_text = "Your pricing effectively captures value hunters. The deep discount suggests a volume-play strategy."
                else:
                    anchor_text = "Individual prices are set close to the bundle price. This signals 'Premium Quality' across the board, extracting maximum value from loyalists while nudging high-WTP users to the bundle."
                    segment_text = "The strategy focuses on **Premium Extraction**. You are willing to sacrifice some volume to maintain high margins per unit."

                st.markdown(f"""
                <div class="strategy-box" style="background: white;">
                    <div style="margin-bottom: 15px;">
                        <strong>1. The Decoy Effect & Anchoring</strong><br>
                        <span style="color:#475569; font-size:0.95rem;">{anchor_text}</span>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <strong>2. Market Segmentation</strong><br>
                        <span style="color:#475569; font-size:0.95rem;">{segment_text}</span>
                    </div>
                    <div>
                        <strong>3. Cross-Sell Opportunity</strong><br>
                        <span style="color:#475569; font-size:0.95rem;">
                        Marketing should highlight the <b>₹{(sum_indiv_opt - bundle_price):,.0f}</b> savings. 
                        This calculates to an average item price of <b>₹{(bundle_price/len(products)):,.0f}</b> in the bundle.
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                # Revenue Split Chart
                rev_bundle = customer_df[customer_df['Decision'] == 'Bundle']['Revenue'].sum()
                rev_indiv = customer_df[customer_df['Decision'] == 'Individual']['Revenue'].sum()
                
                fig_donut = px.pie(
                    names=['Bundle Revenue', 'Individual Sales'],
                    values=[rev_bundle, rev_indiv],
                    hole=0.6,
                    color_discrete_sequence=['#4f46e5', '#cbd5e1'],
                    title="Revenue Composition"
                )
                fig_donut.update_layout(showlegend=True, height=250, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_donut, use_container_width=True)

        # TAB 2: DATA TABLE
        with tab2:
            st.markdown("#### Detailed Purchase Decisions")
            st.dataframe(
                customer_df,
                column_config={
                    "Customer ID": st.column_config.NumberColumn(format="#%d"),
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Consumer Surplus": st.column_config.ProgressColumn(format="₹%d", max_value=int(customer_df['Consumer Surplus'].max())),
                    "Decision": st.column_config.Column(width="small"),
                    "Items Bought": st.column_config.Column(width="medium"),
                },
                use_container_width=True,
                height=500,
                hide_index=True
            )

        # TAB 3: DEMAND CHART
        with tab3:
            st.markdown("#### Bundle Price Sensitivity")
            st.caption("How does demand change if we raise or lower the Bundle Price (keeping individual prices constant)?")
            
            demand_data = generate_demand_curve(df, products, opt_prices)
            
            fig = px.area(demand_data, x="Price", y="Demand")
            fig.add_vline(x=bundle_price, line_dash="dash", line_color="#10b981", annotation_text="Optimal Price")
            
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748b"),
                hovermode="x unified",
                xaxis=dict(title="Bundle Price (₹)", showgrid=False),
                yaxis=dict(title="Number of Buyers", showgrid=True, gridcolor='#e2e8f0')
            )
            fig.update_traces(line_color='#6366f1', fillcolor='rgba(99, 102, 241, 0.1)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
