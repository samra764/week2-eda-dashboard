# ============================================================
# WEEK 2 - EDA DASHBOARD (ULTIMATE VERSION)
# Dataset: Telco Customer Churn
# Author: Samra Haseeb | Dawoodtech Internship
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Telco Churn EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BACKGROUND IMAGE + CUSTOM STYLING
# ============================================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image("assets/background.jpg")

if img_base64:
    bg_css = f"""
    background-image: linear-gradient(rgba(8, 12, 25, 0.82), rgba(8, 12, 25, 0.82)),
                       url(data:image/jpg;base64,{img_base64});
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    """
else:
    bg_css = "background: radial-gradient(circle at top, #11203f, #050a16 70%);"

st.markdown(f"""
<style>
.stApp {{
    {bg_css}
}}

[data-testid="stMetric"] {{
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 12px;
    padding: 15px;
}}

[data-testid="stMetricValue"] {{
    color: #ffffff !important;
}}

[data-testid="stMetricLabel"] {{
    color: #cfd8e3 !important;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(8, 12, 25, 0.95);
}}

[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}

h1, h2, h3, h4, p, label, span {{
    color: #ffffff !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
}}

.stTabs [data-baseweb="tab"] p {{
    color: #ffffff !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: rgba(255, 75, 75, 0.3) !important;
}}

[data-testid="stExpander"] {{
    background-color: rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}}

div[data-testid="stDataFrame"] {{
    background-color: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
}}

.stButton > button {{
    background-color: rgba(255, 75, 75, 0.85) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

.stButton > button:hover {{
    background-color: rgba(255, 75, 75, 1) !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}}

.stButton > button p {{
    color: #ffffff !important;
}}

.risk-box {{
    background: linear-gradient(135deg, rgba(255,75,75,0.2), rgba(255,75,75,0.05));
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}

.compare-card {{
    background-color: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}}
</style>
""", unsafe_allow_html=True)

if not img_base64:
    st.sidebar.warning("⚠️ background.jpg not found in /assets — using fallback dark background.")

# ============================================================
# LOAD & CLEAN DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if df['SeniorCitizen'].dtype != 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

df = load_data()

COLOR_MAP = {'Yes': '#FF4B4B', 'No': '#3DAEFF'}

# ============================================================
# DARK THEME HELPER FOR ALL PLOTLY CHARTS
# ============================================================
def style_dark(fig):
    fig.update_layout(
        paper_bgcolor='rgba(15, 20, 35, 0.85)',
        plot_bgcolor='rgba(15, 20, 35, 0.85)',
        font=dict(color='white', size=13),
        title_font=dict(color='white', size=16),
        legend=dict(font=dict(color='white')),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        margin=dict(t=60, b=40, l=40, r=20)
    )
    fig.update_traces(
        textfont=dict(color='white', size=13),
        selector=dict(type='pie')
    )
    return fig

# ============================================================
# CHART CAROUSEL HELPER
# ============================================================
def chart_carousel(carousel_key, figures):
    state_key = f"carousel_idx_{carousel_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    total = len(figures)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])

    with nav_col1:
        if st.button("⬅ Prev", key=f"prev_{carousel_key}", use_container_width=True):
            st.session_state[state_key] = (st.session_state[state_key] - 1) % total

    with nav_col3:
        if st.button("Next ➡", key=f"next_{carousel_key}", use_container_width=True):
            st.session_state[state_key] = (st.session_state[state_key] + 1) % total

    current_idx = st.session_state[state_key]
    chart_type, fig = figures[current_idx]

    with nav_col2:
        st.markdown(
            f"<p style='text-align:center; color:#cfd8e3; margin-top:8px;'>"
            f"Chart {current_idx + 1} of {total}</p>",
            unsafe_allow_html=True
        )

    if chart_type == "plotly":
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{carousel_key}_{current_idx}")
    else:
        st.pyplot(fig, use_container_width=True)

    dots = "".join(["⬤ " if i == current_idx else "○ " for i in range(total)])
    st.markdown(
        f"<p style='text-align:center; color:#FF4B4B; letter-spacing:4px;'>{dots}</p>",
        unsafe_allow_html=True
    )

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("🔧 Filter Options")
st.sidebar.markdown("---")

churn_filter = st.sidebar.multiselect(
    "Churn Status", options=df['Churn'].unique(), default=df['Churn'].unique()
)
contract_filter = st.sidebar.multiselect(
    "Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique()
)
internet_filter = st.sidebar.multiselect(
    "Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique()
)

# NEW: Tenure range slider in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**📅 Tenure Range (months)**")
tenure_range = st.sidebar.slider(
    "",
    min_value=int(df['tenure'].min()),
    max_value=int(df['tenure'].max()),
    value=(0, int(df['tenure'].max()))
)

# NEW: Monthly charges range slider in sidebar
st.sidebar.markdown("**💰 Monthly Charges Range ($)**")
charge_range = st.sidebar.slider(
    " ",
    min_value=float(df['MonthlyCharges'].min()),
    max_value=float(df['MonthlyCharges'].max()),
    value=(float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()))
)

filtered_df = df[
    (df['Churn'].isin(churn_filter)) &
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter)) &
    (df['tenure'].between(tenure_range[0], tenure_range[1])) &
    (df['MonthlyCharges'].between(charge_range[0], charge_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption("Built by Samra Haseeb | Dawoodtech Internship")

# ============================================================
# COMPUTED METRICS (used across multiple tabs)
# ============================================================
churn_rate_val = round(filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100, 1)
retention_rate = round(100 - churn_rate_val, 1)
avg_charge_churn = round(filtered_df[filtered_df['Churn'] == 'Yes']['MonthlyCharges'].mean(), 2) if (filtered_df['Churn'] == 'Yes').any() else 0
avg_charge_retained = round(filtered_df[filtered_df['Churn'] == 'No']['MonthlyCharges'].mean(), 2) if (filtered_df['Churn'] == 'No').any() else 0
mtm_pct = round(
    filtered_df[(filtered_df['Contract'] == 'Month-to-month') & (filtered_df['Churn'] == 'Yes')].shape[0]
    / max(filtered_df[filtered_df['Churn'] == 'Yes'].shape[0], 1) * 100, 1
)
completeness = round(
    (1 - filtered_df.isnull().sum().sum() / (filtered_df.shape[0] * filtered_df.shape[1])) * 100, 1
)

# ============================================================
# HEADER
# ============================================================
st.title("📊 Telco Customer Churn — EDA Dashboard")
st.markdown("Interactive exploratory data analysis built with Streamlit | Dawoodtech Internship Week 2")

# ============================================================
# AUTO-ROTATING INSIGHT CAROUSEL
# ============================================================
carousel_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{ margin: 0; padding: 0; background: transparent; }}
.carousel {{
    position: relative;
    width: 100%;
    height: 140px;
    border-radius: 15px;
    overflow: hidden;
    font-family: 'Source Sans Pro', sans-serif;
}}
.carousel-slide {{
    position: absolute;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
    font-size: 20px;
    font-weight: 600;
    border-radius: 15px;
    opacity: 0;
    animation: fade 12s infinite;
    padding: 20px;
    box-sizing: border-box;
}}
.slide1 {{ background: linear-gradient(135deg, #1e3c72, #2a5298); animation-delay: 0s; }}
.slide2 {{ background: linear-gradient(135deg, #11998e, #38ef7d); animation-delay: 4s; }}
.slide3 {{ background: linear-gradient(135deg, #cb2d3e, #ef473a); animation-delay: 8s; }}
@keyframes fade {{
    0%   {{ opacity: 0; }}
    3%   {{ opacity: 1; }}
    30%  {{ opacity: 1; }}
    36%  {{ opacity: 0; }}
    100% {{ opacity: 0; }}
}}
</style>
</head>
<body>
<div class="carousel">
    <div class="carousel-slide slide1">📉 Current Churn Rate: {churn_rate_val}% of filtered customers</div>
    <div class="carousel-slide slide2">💰 Churned customers pay ${avg_charge_churn} vs ${avg_charge_retained} for retained customers</div>
    <div class="carousel-slide slide3">📅 {mtm_pct}% of churned customers were on month-to-month contracts</div>
</div>
</body>
</html>
"""
components.html(carousel_html, height=150)

# ============================================================
# MAIN TABS
# ============================================================
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "1️⃣ Overview",
    "2️⃣ Data Cleaning",
    "3️⃣ EDA Charts",
    "4️⃣ Custom Chart",
    "5️⃣ Churn Analyser",
    "6️⃣ Insights"
])

# ============================================================
# TAB 1 — DATASET OVERVIEW
# ============================================================
with main_tab1:
    st.subheader("Dataset Overview")

    # Top metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", filtered_df.shape[0])
    col2.metric("Total Features", filtered_df.shape[1])
    col3.metric("Churn Rate", f"{churn_rate_val}%")
    col4.metric("Retention Rate", f"{retention_rate}%")
    col5.metric("Avg Monthly Charges", f"${round(filtered_df['MonthlyCharges'].mean(), 2)}")

    st.markdown("---")

    # NEW: KPI Progress Bars
    st.markdown("### 🎯 KPI Progress Tracker")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Churn Rate Target: Stay below 25%**")
        progress_val = min(int(churn_rate_val), 100)
        color_label = "🔴 ABOVE TARGET" if churn_rate_val > 25 else "🟢 ON TARGET"
        st.progress(progress_val, text=f"Current: {churn_rate_val}% — {color_label}")

    with kpi2:
        st.markdown("**Retention Rate Target: Above 75%**")
        st.progress(min(int(retention_rate), 100), text=f"Current: {retention_rate}%")

    with kpi3:
        st.markdown("**Data Completeness**")
        st.progress(min(int(completeness), 100), text=f"{completeness}% complete")

    st.markdown("---")

    # NEW: Gauge chart + Funnel chart side by side
    gauge_col, funnel_col = st.columns(2)

    with gauge_col:
        st.markdown("### 📊 Churn Rate Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_rate_val,
            delta={
                'reference': 25,
                'increasing': {'color': "#FF4B4B"},
                'decreasing': {'color': "#38ef7d"}
            },
            title={'text': "Churn Rate %", 'font': {'color': 'white', 'size': 16}},
            number={'font': {'color': 'white', 'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': "#FF4B4B"},
                'bgcolor': 'rgba(15,20,35,0)',
                'steps': [
                    {'range': [0, 25], 'color': "rgba(61,174,255,0.25)"},
                    {'range': [25, 50], 'color': "rgba(255,165,0,0.25)"},
                    {'range': [50, 100], 'color': "rgba(255,75,75,0.25)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 25
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(15,20,35,0.85)',
            font=dict(color='white'),
            height=300,
            margin=dict(t=40, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with funnel_col:
        st.markdown("### 🔽 Customer Retention Funnel")
        total_customers = len(filtered_df)
        active_long = len(filtered_df[filtered_df['tenure'] > 12])
        at_risk = len(filtered_df[
            (filtered_df['tenure'] <= 12) &
            (filtered_df['MonthlyCharges'] > 70)
        ])
        churned_total = len(filtered_df[filtered_df['Churn'] == 'Yes'])

        fig_funnel = go.Figure(go.Funnel(
            y=["Total Customers", "Tenured > 1yr", "At Risk (New + High Charge)", "Churned"],
            x=[total_customers, active_long, at_risk, churned_total],
            textinfo="value+percent initial",
            textfont=dict(color='white', size=13),
            marker=dict(color=["#3DAEFF", "#38ef7d", "#FFA500", "#FF4B4B"])
        ))
        fig_funnel.update_layout(
            paper_bgcolor='rgba(15,20,35,0.85)',
            plot_bgcolor='rgba(15,20,35,0.85)',
            font=dict(color='white', size=13),
            height=300,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # NEW: Live Search
    st.markdown("### 🔎 Live Data Search")
    search = st.text_input("Search any value across the dataset (e.g. 'Month-to-month', 'Yes', 'Fiber optic')")

    if search:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        result_df = filtered_df[mask]
        st.caption(f"Found **{mask.sum()}** matching rows out of {len(filtered_df)}")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    else:
        with st.expander("📋 View Sample Data (first 10 rows)"):
            st.dataframe(filtered_df.head(10), use_container_width=True, hide_index=True)

    with st.expander("📌 View Data Types"):
        dtype_df = filtered_df.dtypes.reset_index()
        dtype_df.columns = ['Column', 'Data Type']
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with st.expander("📊 View Summary Statistics"):
        st.dataframe(filtered_df.describe(), use_container_width=True)

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_telco_data.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 2 — DATA CLEANING
# ============================================================
with main_tab2:
    st.subheader("Data Cleaning Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", filtered_df.isnull().sum().sum())
    col2.metric("Duplicate Rows", filtered_df.duplicated().sum())
    col3.metric("Columns Cleaned", "2")

    with st.expander("🔍 View Missing Value Breakdown"):
        missing = filtered_df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing %'] = round(missing['Missing Count'] / len(filtered_df) * 100, 2)
        st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown("#### ✅ Cleaning Steps Applied")
    st.success("✔ TotalCharges converted from object to numeric (float)")
    st.success("✔ SeniorCitizen converted from 0/1 to Yes/No for readability")
    st.success("✔ customerID column removed (not useful for analysis)")
    st.success("✔ Missing TotalCharges filled with median value")
    st.success("✔ Duplicate rows checked and confirmed clean")

    # NEW: Churned vs Retained Comparison Table
    st.markdown("---")
    st.markdown("### 📋 Churned vs Retained — Side by Side Comparison")

    churned_stats = filtered_df[filtered_df['Churn'] == 'Yes'][
        ['tenure', 'MonthlyCharges', 'TotalCharges']
    ].mean()
    retained_stats = filtered_df[filtered_df['Churn'] == 'No'][
        ['tenure', 'MonthlyCharges', 'TotalCharges']
    ].mean()

    compare_df = pd.DataFrame({
        'Metric': ['Avg Tenure (months)', 'Avg Monthly Charges ($)', 'Avg Total Charges ($)'],
        '✅ Retained Customers': [
            round(retained_stats['tenure'], 1),
            round(retained_stats['MonthlyCharges'], 2),
            round(retained_stats['TotalCharges'], 2)
        ],
        '🔴 Churned Customers': [
            round(churned_stats['tenure'], 1),
            round(churned_stats['MonthlyCharges'], 2),
            round(churned_stats['TotalCharges'], 2)
        ],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # Visual comparison bars
    st.markdown("#### 📊 Visual Comparison")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        fig_comp1 = go.Figure(go.Bar(
            x=['Retained', 'Churned'],
            y=[retained_stats['MonthlyCharges'], churned_stats['MonthlyCharges']],
            marker_color=['#3DAEFF', '#FF4B4B'],
            text=[f"${retained_stats['MonthlyCharges']:.2f}", f"${churned_stats['MonthlyCharges']:.2f}"],
            textposition='auto',
            textfont=dict(color='white')
        ))
        fig_comp1.update_layout(title='Avg Monthly Charges Comparison')
        st.plotly_chart(style_dark(fig_comp1), use_container_width=True)

    with comp_col2:
        fig_comp2 = go.Figure(go.Bar(
            x=['Retained', 'Churned'],
            y=[retained_stats['tenure'], churned_stats['tenure']],
            marker_color=['#3DAEFF', '#FF4B4B'],
            text=[f"{retained_stats['tenure']:.1f} mo", f"{churned_stats['tenure']:.1f} mo"],
            textposition='auto',
            textfont=dict(color='white')
        ))
        fig_comp2.update_layout(title='Avg Tenure Comparison')
        st.plotly_chart(style_dark(fig_comp2), use_container_width=True)

# ============================================================
# TAB 3 — EDA CHARTS
# ============================================================
with main_tab3:
    st.subheader("Exploratory Data Analysis")

    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
        ["📊 Churn Overview", "📈 Distributions", "🔗 Relationships", "🔥 Correlation"]
    )

    with chart_tab1:
        churn_counts = filtered_df['Churn'].value_counts()
        fig1 = px.bar(
            x=churn_counts.index, y=churn_counts.values,
            color=churn_counts.index, labels={'x': 'Churn', 'y': 'Count'},
            title='Churn Count', color_discrete_map=COLOR_MAP
        )
        fig2 = px.pie(
            filtered_df, names='Churn', title='Churn Percentage',
            color='Churn', color_discrete_map=COLOR_MAP
        )
        internet_churn = filtered_df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
        fig_inet = px.bar(
            internet_churn, x='InternetService', y='Count', color='Churn',
            barmode='group', title='Churn by Internet Service',
            color_discrete_map=COLOR_MAP
        )
        chart_carousel("churn_overview", [
            ("plotly", style_dark(fig1)),
            ("plotly", style_dark(fig2)),
            ("plotly", style_dark(fig_inet))
        ])

    with chart_tab2:
        fig3 = px.histogram(
            filtered_df, x='MonthlyCharges', color='Churn', nbins=30,
            title='Monthly Charges Distribution', color_discrete_map=COLOR_MAP
        )
        fig4 = px.histogram(
            filtered_df, x='tenure', color='Churn', nbins=24,
            title='Tenure Distribution', color_discrete_map=COLOR_MAP
        )
        fig_total = px.histogram(
            filtered_df, x='TotalCharges', color='Churn', nbins=30,
            title='Total Charges Distribution', color_discrete_map=COLOR_MAP
        )
        chart_carousel("distributions", [
            ("plotly", style_dark(fig3)),
            ("plotly", style_dark(fig4)),
            ("plotly", style_dark(fig_total))
        ])

    with chart_tab3:
        fig5 = px.box(
            filtered_df, x='Churn', y='MonthlyCharges', color='Churn',
            title='Monthly Charges by Churn Status', color_discrete_map=COLOR_MAP
        )
        contract_churn = filtered_df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig6 = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn', barmode='group',
            title='Churn by Contract Type', color_discrete_map=COLOR_MAP
        )
        fig8 = px.scatter(
            filtered_df, x='tenure', y='MonthlyCharges', color='Churn',
            title='Tenure vs Monthly Charges', color_discrete_map=COLOR_MAP, opacity=0.5
        )
        fig_senior = filtered_df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count')
        fig_sc = px.bar(
            fig_senior, x='SeniorCitizen', y='Count', color='Churn',
            barmode='group', title='Churn by Senior Citizen Status',
            color_discrete_map=COLOR_MAP
        )
        chart_carousel("relationships", [
            ("plotly", style_dark(fig5)),
            ("plotly", style_dark(fig6)),
            ("plotly", style_dark(fig8)),
            ("plotly", style_dark(fig_sc))
        ])

    with chart_tab4:
        numeric_df = filtered_df.select_dtypes(include=np.number)
        fig7, ax = plt.subplots(figsize=(8, 4))
        fig7.patch.set_facecolor('#0a0f1e')
        ax.set_facecolor('#0a0f1e')
        heatmap = sns.heatmap(
            numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, annot_kws={"color": "white", "fontsize": 11},
            cbar_kws={"label": ""}
        )
        ax.set_title("Correlation Heatmap — Numeric Features", color="white", fontsize=13)
        ax.tick_params(colors='white')
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig7, use_container_width=True)

# ============================================================
# TAB 4 — CUSTOM INTERACTIVE CHART
# ============================================================
with main_tab4:
    st.subheader("🎨 Build Your Own Visualization")
    st.markdown("Select any columns and chart type below to generate a custom chart instantly.")

    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("Select X-axis", options=filtered_df.columns)
    with col2:
        y_col = st.selectbox("Select Y-axis", options=filtered_df.select_dtypes(include=np.number).columns)
    with col3:
        chart_type = st.selectbox("Select Chart Type", options=["Scatter", "Bar", "Box", "Histogram", "Violin"])

    color_by = st.selectbox("Color By", options=['Churn', 'Contract', 'InternetService', 'SeniorCitizen'])

    if chart_type == "Scatter":
        fig9 = px.scatter(filtered_df, x=x_col, y=y_col, color=color_by)
    elif chart_type == "Bar":
        fig9 = px.bar(filtered_df, x=x_col, y=y_col, color=color_by)
    elif chart_type == "Box":
        fig9 = px.box(filtered_df, x=x_col, y=y_col, color=color_by)
    elif chart_type == "Violin":
        fig9 = px.violin(filtered_df, x=x_col, y=y_col, color=color_by, box=True)
    else:
        fig9 = px.histogram(filtered_df, x=x_col, color=color_by)

    st.plotly_chart(style_dark(fig9), use_container_width=True)

# ============================================================
# TAB 5 — CHURN ANALYSER (NEW)
# ============================================================
with main_tab5:
    st.subheader("🤖 Churn Risk Analyser")
    st.markdown("Two powerful tools to understand and predict churn risk.")

    analyser_tab1, analyser_tab2 = st.tabs(["🔮 Churn Risk Estimator", "📊 Segment Analysis"])

    # --- Churn Risk Estimator ---
    with analyser_tab1:
        st.markdown("### Enter Customer Profile to Estimate Churn Risk")
        st.markdown("Adjust the sliders to match a customer's profile and instantly see their estimated churn risk.")

        est_col1, est_col2, est_col3 = st.columns(3)

        with est_col1:
            tenure_input = st.slider("📅 Tenure (months)", 0, 72, 12)
            senior_input = st.selectbox("👤 Senior Citizen", ["No", "Yes"])

        with est_col2:
            charge_input = st.slider("💰 Monthly Charges ($)", 20, 120, 65)
            internet_input = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

        with est_col3:
            contract_input = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"])
            support_input = st.selectbox("🛠 Tech Support", ["No", "Yes"])

        # Rule-based risk score using EDA insights
        risk = 0
        reasons = []

        if tenure_input < 12:
            risk += 35
            reasons.append("🔴 New customer (tenure < 12 months) — highest risk group")
        elif tenure_input < 24:
            risk += 18
            reasons.append("🟡 Moderate tenure (12–24 months) — medium risk")
        else:
            reasons.append("🟢 Long-term customer — lower risk")

        if charge_input > 70:
            risk += 28
            reasons.append("🔴 High monthly charges (>$70) — strongly linked to churn")
        elif charge_input > 50:
            risk += 12
            reasons.append("🟡 Moderate monthly charges ($50–$70)")
        else:
            reasons.append("🟢 Low monthly charges — below average churn threshold")

        if contract_input == "Month-to-month":
            risk += 27
            reasons.append("🔴 Month-to-month contract — highest churn contract type")
        elif contract_input == "One year":
            risk += 8
            reasons.append("🟡 One-year contract — moderate risk")
        else:
            reasons.append("🟢 Two-year contract — lowest churn contract type")

        if internet_input == "Fiber optic":
            risk += 7
            reasons.append("🟡 Fiber optic users churn more than DSL users")

        if senior_input == "Yes":
            risk += 5
            reasons.append("🟡 Senior citizens show slightly higher churn tendency")

        if support_input == "No":
            risk += 5
            reasons.append("🟡 No tech support — associated with higher churn")

        risk = min(risk, 100)

        # Display result
        st.markdown("---")
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            if risk > 60:
                risk_label = "🔴 HIGH RISK"
                risk_color = "#FF4B4B"
            elif risk > 35:
                risk_label = "🟡 MEDIUM RISK"
                risk_color = "#FFA500"
            else:
                risk_label = "🟢 LOW RISK"
                risk_color = "#38ef7d"

            st.markdown(
                f"""
                <div style='background:rgba(15,20,35,0.9); border:2px solid {risk_color};
                border-radius:15px; padding:25px; text-align:center;'>
                    <p style='font-size:18px; color:#cfd8e3; margin-bottom:5px;'>Estimated Churn Risk</p>
                    <p style='font-size:52px; font-weight:800; color:{risk_color}; margin:0;'>{risk}%</p>
                    <p style='font-size:18px; color:{risk_color}; font-weight:600;'>{risk_label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(risk)

        with result_col2:
            st.markdown("#### 📋 Risk Factor Breakdown")
            for reason in reasons:
                st.markdown(f"- {reason}")

            st.markdown("---")
            st.markdown("#### 💡 Recommended Actions")
            if risk > 60:
                st.error("⚡ **Immediate action required!**")
                st.error("• Offer a discount or loyalty reward immediately")
                st.error("• Propose a contract upgrade with incentives")
                st.error("• Assign a dedicated support representative")
            elif risk > 35:
                st.warning("• Monitor this customer closely")
                st.warning("• Send a proactive satisfaction survey")
                st.warning("• Offer a tech support bundle")
            else:
                st.success("• Customer appears stable — maintain quality service")
                st.success("• Consider a renewal reminder before contract ends")

    # --- Segment Analysis ---
    with analyser_tab2:
        st.markdown("### 📊 Churn by Customer Segment")

        seg_col = st.selectbox(
            "Select Segment to Analyse",
            options=['Contract', 'InternetService', 'SeniorCitizen', 'PaymentMethod']
        )

        if seg_col in filtered_df.columns:
            seg_data = filtered_df.groupby([seg_col, 'Churn']).size().reset_index(name='Count')
            seg_total = filtered_df.groupby(seg_col).size().reset_index(name='Total')
            seg_merged = seg_data.merge(seg_total, on=seg_col)
            seg_merged['Churn Rate %'] = round(seg_merged['Count'] / seg_merged['Total'] * 100, 1)

            seg_churned = seg_merged[seg_merged['Churn'] == 'Yes'].sort_values('Churn Rate %', ascending=False)

            fig_seg = px.bar(
                seg_churned, x=seg_col, y='Churn Rate %',
                color='Churn Rate %',
                color_continuous_scale=['#3DAEFF', '#FFA500', '#FF4B4B'],
                title=f'Churn Rate % by {seg_col}',
                text='Churn Rate %'
            )
            fig_seg.update_traces(texttemplate='%{text}%', textposition='outside', textfont_color='white')
            st.plotly_chart(style_dark(fig_seg), use_container_width=True)

            st.dataframe(seg_churned[[seg_col, 'Count', 'Total', 'Churn Rate %']],
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"Column '{seg_col}' not found in filtered data.")

# ============================================================
# TAB 6 — INSIGHTS
# ============================================================
with main_tab6:
    st.subheader("Key Insights & Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Key Findings")
        st.info("📌 Overall churn rate is approximately **26%**")
        st.info("📌 **Month-to-month** contract customers churn the most (~43%)")
        st.info("📌 Churned customers pay **higher monthly charges** on average")
        st.info("📌 **Fiber Optic** users churn more than DSL users")
        st.info("📌 Customers in their **first 12 months** are at highest risk")
        st.info("📌 Customers **without tech support** churn significantly more")

    with col2:
        st.markdown("#### ✅ Recommendations")
        st.success("💡 Offer **loyalty discounts** to new customers (0–12 months)")
        st.success("💡 Promote **annual or 2-year contracts** with incentives")
        st.success("💡 Investigate **Fiber Optic** pricing or service quality issues")
        st.success("💡 Add **tech support bundles** to reduce at-risk customer churn")
        st.success("💡 Target **senior citizens** with special retention programs")
        st.success("💡 Launch **early tenure campaigns** for customers in months 1–6")

    st.markdown("---")
    st.markdown("#### 📈 Trend Summary")
    st.warning(
        "Customers who churn tend to share three common traits: "
        "shorter tenure + higher monthly charges + month-to-month contracts. "
        "Addressing any one of these three significantly reduces churn probability."
    )

    # Dynamic insight based on filtered data
    st.markdown("---")
    st.markdown("#### 📊 Live Stats Based on Current Filters")
    live_col1, live_col2, live_col3, live_col4 = st.columns(4)
    live_col1.metric("Filtered Records", len(filtered_df))
    live_col2.metric("Churn Count", len(filtered_df[filtered_df['Churn'] == 'Yes']))
    live_col3.metric("Avg Churned Charge", f"${avg_charge_churn}")
    live_col4.metric("Avg Retained Charge", f"${avg_charge_retained}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<center>Samra Haseeb</center>",
    unsafe_allow_html=True
)
