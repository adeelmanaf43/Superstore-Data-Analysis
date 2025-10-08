# Importing required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for typography and style
st.markdown("""
    <style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Roboto', sans-serif;
        color: #2c3e50;
    }

    /* Main container */
    .main {
        padding: 0rem 1rem;
    }

    /* Headers */
    h1 {
        color: #1f77b4;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    h2 {
        color: #34495e;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }
    h3, h4 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    /* Paragraphs */
    p {
        font-size: 15px;
        line-height: 1.6;
        color: #444;
    }

    /* KPI metric cards */
    .stMetric {
        background-color: #f9f9fb;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric label {
        font-size: 14px;
        font-weight: 600;
        color: #555;
    }
    .stMetric div {
        font-size: 20px;
        font-weight: 700;
        color: #1f77b4;
    }

    /* Insight / zone boxes */
    .insight-box, .safe-zone, .danger-zone {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        color: #fff;
    }
    .insight-box {
        background: linear-gradient(135deg, #f0f0f0 0%, #eddfdf 100%);
    }
    .safe-zone {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .danger-zone {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .insight-box h3, .insight-box h4,
    .safe-zone h4, .danger-zone h4 {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.25);
    }
    .insight-box p, .safe-zone p, .danger-zone p {
        font-size: 15px;
        line-height: 1.6;
        margin: 6px 0;
    }
    .insight-box strong, .safe-zone strong, .danger-zone strong {
        color: #f5f0f0;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 15px;
        background-color: #f9f9fb;
        border-radius: 10px;
        margin-top: 30px;
    }
    .footer p {
        color: #7f8c8d;
        font-size: 14px;
    }
    .footer a {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 600;
    }
    .footer a:hover {
        color: #145a86;
    }
    strong{
        color: black
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading


@st.cache_data
def load_data(ttl=600):
    """Load and preprocess the dataset"""
    df = pd.read_csv("superstore_cleaned.csv")

    # Convert date columns
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df


# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "⚠️ Data file not found! Please ensure 'superstore_cleaned.csv' is in the same directory")
    st.stop()

# ===== SIDEBAR =====
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/business-report.png", width=150)
st.sidebar.title("🎯 Dashboard Controls")
st.sidebar.markdown("---")


# Date range filter
st.sidebar.subheader("📅 Date Range")
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# Apply date filter
if len(date_range) == 2:
    mask = (df['Order Date'] >= pd.to_datetime(date_range[0])) & (
        df['Order Date'] <= pd.to_datetime(date_range[1]))
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()


# Category filter
st.sidebar.subheader("🏷️ Category")
categories = ['All'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", categories)
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

# Region filter
st.sidebar.subheader("🗺️ Region")
regions = ['All'] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

# Segmenet filter
st.sidebar.subheader("👥Customer Segment")
segments = ['All'] + sorted(df['Segment'].unique().tolist())
selected_segment = st.sidebar.selectbox('Select Segment', segments)
if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

# ===== MAIN DASHBOARD =====
st.title("📊 Sales Analytics Dashboard")
st.markdown("### Superstore Profitability Analysis & Business Intelligence")
st.markdown("---")

# ===== KPI METRICS =====
st.subheader("🎯 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
profit_margin = (total_profit / total_sales * 100)
total_orders = filtered_df['Order ID'].nunique()
unprofitable_rate = (filtered_df[filtered_df['Profit'] < 0].shape[0] /
                     len(filtered_df) * 100) if len(filtered_df) > 0 else 0

with col1:
    st.metric("💰 Total Sales", f"${total_sales:,.0f}")

with col2:
    st.metric("📈 Total Profit", f"${total_profit:,.0f}")

with col3:
    st.metric("💹 Profit Margin", f"{profit_margin:,.2f}")
with col4:
    st.metric("📦 Total Orders", f"{total_orders:,}")
with col5:
    color = "inverse" if unprofitable_rate > 15 else "off"
    st.metric("⚠️ Unprofitable",
              f"{unprofitable_rate:.1f}%", delta=None, delta_color=color)

st.markdown("---")


# ===== TAB NAVIGATION =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "💸 Discount Analysis",
    "👥 Customer Insights",
    "📦 Product Performance",
    "Regional Analysis",
    "🎯 Key Findings"
])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.header("Business Overview")
    col1, col2 = st.columns(2)
    with col1:
        # Sales and Profit Trend
        monthly_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        monthly_data['Order Date'] = monthly_data['Order Date'].dt.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data['Order Date'],
            y=monthly_data['Sales'],
            name='Sales',
            line=dict(color="#3498db", width=3),
            fill='tozeroy'
        ))
        fig.add_trace(go.Scatter(
            x=monthly_data['Order Date'],
            y=monthly_data['Profit'],
            name='Profit',
            line=dict(color='#2ecc71', width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            title='Sales & Profit Trends Over Time',
            xaxis_title='Data',
            yaxis_title='Sales ($)',
            yaxis2=dict(title='Profit ($)', overlaying='y', side='right'),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Category Performance
        category_data = filtered_df.groupby("Category").agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        category_data['Profit Margin %'] = (
            category_data['Profit'] / category_data['Sales'] * 100).round(2)
        fig = px.bar(
            category_data, x='Category', y='Sales', color='Profit Margin %',
            color_continuous_scale='RdYlGn', title='Category Performance',
            text='Sales'
        )
        fig.update_traces(
            texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_yaxes(
            range=[0, 1000000],
            tickvals=[0, 200000, 400000, 600000, 800000, 1000000],
            ticktext=['0', '200K', '400K', '600K', '800K', '1,000K']
        )
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Category-Region Heatmap
    st.subheader("🗺️ Category-Region Profit Margin Heatmap")
    heatmap_data = filtered_df.groupby(['Category', 'Region']).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    heatmap_data['Profit Margin %'] = (
        heatmap_data['Profit'] / heatmap_data['Sales'] * 100).round(2)
    pivot_data = heatmap_data.pivot(
        index='Category', columns='Region', values='Profit Margin %')
    fig2 = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=pivot_data.values,
        texttemplate='%{text:.1f}%',
        textfont={'size': 14},
        colorbar=dict(title='Margin %')
    ))
    fig2.update_layout(
        title='Profit Margin by Category and Region', height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig2, use_container_width=True)


# ===== TAB 2: DISCOUNT ANALYSIS =====
with tab2:
    st.header("💸 Discont Impact Analysis")
    st.subheader("🚨 The Profit Cliff: Discount Profitability Analysis")
    discount_bins = pd.cut(
        filtered_df['Discount'],
        bins=np.arange(0, 0.85, 0.05),
        labels=[f'{i}-{i+5}%' for i in range(0, 80, 5)]
    )
    cliff_data = filtered_df.groupby(discount_bins).agg({
        'Profit': ['sum', 'count', lambda x: (x > 0).mean()*100],
        'Sales': 'sum'
    }).reset_index()
    cliff_data.columns = ['Discount Range', 'Total Profit',
                          'Order Count', 'Profit Rate %', 'Total Sales']
    cliff_data = cliff_data[cliff_data['Order Count'] > 0]
    cliff_data['Order %'] = (
        cliff_data['Order Count'] / cliff_data['Order Count'].sum() * 100).round(1)
    fig = go.Figure()
    colors = ['green' if x > 80 else 'orange' if x >
              50 else 'red' for x in cliff_data['Profit Rate %']]
    fig.add_trace(go.Bar(
        x=cliff_data['Discount Range'], y=cliff_data['Profit Rate %'],
        marker_color=colors,
        text=cliff_data['Profit Rate %'].round(1),
        textposition='outside',
        name='Profitability Rate',
        hovertemplate='<b>%{x}</b><br>Profit Rate: %{y:.1f}%<br>Orders %:%{customdata:.1f}%<extra></extra>',
        customdata=cliff_data['Order %']
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='red',
                  annotation_text='50% Survival Line')
    fig.update_layout(
        title='The Profit Cliff: Profitability by Discount Range',
        xaxis_title='Discount Range',
        yaxis_title='Profitability Rate (%)',
        template='plotly_white',
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    danger_zone = filtered_df[filtered_df['Discount'] >= 0.25]
    safe_zone = filtered_df.loc[filtered_df['Discount'] < 0.25, :]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class = "insight-box">
        <h4 style = 'color: #444;'>🟢 Safe Zone (<25% discount) </h4>
        <p><strong style = 'color: #444'> Orders:</strong> {:,} </p>
        <p><strong style = 'color: #444'>Avg Profit: </strong> ${:.2f}
        <p><strong style = 'color: #444'>Profitability Rate: </strong>{:.1f}%</p>
        </div>
""".format(len(safe_zone), safe_zone['Profit'].mean() if len(safe_zone) > 0 else 0,
            (safe_zone['Profit'] > 0).mean() * 100 if len(safe_zone) > 0 else 0),
            unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="insight-box">
            <h4 style = 'color: #444'>🔴 Danger Zone (≥25% discount)</h4>
            <p><strong style = 'color: #444'>Orders:</strong> {:,}</p>
            <p><strong style = 'color: #444'>Avg Profit:</strong> ${:.2f}</p>
            <p><strong style = 'color: #444'>Profitability Rate:</strong> {:.1f}%</p>
            </div>
            """.format(
            len(danger_zone),
            danger_zone['Profit'].mean() if len(danger_zone) > 0 else 0,
            (danger_zone['Profit'] > 0).mean() *
            100 if len(danger_zone) > 0 else 0
        ), unsafe_allow_html=True)

    # Scatter plot
    st.subheader("Discount vs Profitability Scatter")
    fig = px.scatter(filtered_df, x='Discount', y='Profit', color='Category', size='Sales',
                     hover_data=['Product Name', 'Region', 'Segment'],
                     opacity=0.6)
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.add_vline(x=0.25, line_dash='dash', line_color='orange')
    fig.update_xaxes(tickformat='.0%')
    fig.update_layout(template='plotly_white', height=500)
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB3: CUSTOMER INSIGHTS =====
with tab3:
    st.header("👥 Customer Analytics")
    col1, col2 = st.columns(2)
    with col1:
        # Customer segmentation
        customer_data = filtered_df.groupby('Customer ID').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        customer_data.columns = ['Customer ID',
                                 'Lifetime Sales', 'Lifetime Profit', 'Total Orders']
        customer_data = customer_data.sort_values(
            'Lifetime Profit', ascending=False)

        # Customer value pyramid
        customer_data['Cumulative Profit'] = customer_data['Lifetime Profit'].cumsum()
        customer_data['Cumulative %'] = (
            customer_data['Cumulative Profit'] / customer_data['Lifetime Profit'].sum() * 100).round(3)
        customer_data['Customer Rank %'] = (
            np.arange(len(customer_data)) + 1) / len(customer_data) * 100

        customer_data['Segment'] = pd.cut(
            customer_data['Customer Rank %'],
            bins=[0, 20, 50, 100],
            labels=['Top 20%', 'Middle 30%', 'Bottom 50%']
        )
        pyramid = customer_data.groupby('Segment').agg({
            'Customer ID': 'count',
            'Lifetime Profit': 'sum'
        }).reset_index()
        pyramid['Profit %'] = round(
            pyramid['Lifetime Profit'] / pyramid['Lifetime Profit'].sum() * 100, 1)
        fig = px.bar(
            pyramid, x='Segment', y='Profit %', color='Profit %',
            color_continuous_scale='Viridis',
            title='Customer Value pyramid', text='Profit %'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(template='plotly_white', height=400,  yaxis=dict(
            title="Profit Percentage",
            range=[-25, 100],
            tick0=-25,
            dtick=25
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Segment Performance
        segment_data = filtered_df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count'
        }).reset_index()
        segment_data['Profit Margin %'] = (
            segment_data['Profit'] / segment_data['Sales'] * 100)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=segment_data['Segment'], y=segment_data['Sales'],
            name='Sales', marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=segment_data['Segment'], y=segment_data['Profit Margin %'],
            name='Profit Margin%', yaxis='y2', mode='lines+markers',
            marker=dict(size=12, color='darkgreen'),
            line=dict(width=5)
        ))
        fig.update_layout(
            title='Segment Performance',
            yaxis=dict(title='Sales ($)'),
            yaxis2=dict(title='Profit Margin %', overlaying='y', side='right'),
            template='plotly_white', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top customers table
    st.subheader("🏆 Top 10 Customers by Lifetime Value")
    top_customers = filtered_df.groupby(['Customer ID', 'Customer Name']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).reset_index()
    top_customers.columns = [
        'Customer ID', 'Customer Name', 'Lifetime Sales', 'Lifetime Profit', 'Total Orders']
    top_customers = top_customers.sort_values(
        by='Lifetime Profit', ascending=False).head(10)
    st.dataframe(
        top_customers.style.format({
            'Lifetime Sales': '${:,.2f}',
            'Lifetime Profit': '${:,.2f}'
        }).background_gradient(subset=['Lifetime Profit'], cmap='RdYlGn'),
        use_container_width=True
    )


# ===== TAB 4: PRODUCT PERFORMANCE =====
with tab4:
    st.header("📦 Product Performance Analysis")

    # Sub-category performance
    subcat_data = filtered_df.groupby('Sub-Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()

    subcat_data['Profit Margin %'] = round(
        subcat_data['Profit'] / subcat_data['Sales'] * 100, 1)
    subcat_data['Profit'] = subcat_data['Profit'].round(2)
    subcat_data = subcat_data.sort_values('Profit', ascending=True)
    col1, col2 = st.columns(2)
    with col1:
        # Top performers
        top_subcat = subcat_data.tail(10)
        fig = px.bar(
            top_subcat,
            x='Profit', y='Sub-Category', orientation='h',
            color='Profit Margin %', color_continuous_scale='Greens',
            title='🏆 Top 10 Most Profitable Sub-Categories'
        )
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bottom Performers
        bottom_subcat = subcat_data.head(10)
        fig = px.bar(
            bottom_subcat, x='Profit', y='Sub-Category', orientation='h',
            color='Profit', color_continuous_scale='Reds',
            title='⚠️ Bottom 10 Sub-Categories by Profit'
        )
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("📊 Complete Sub-Category Performance")
    subcat_display = subcat_data.sort_values(
        'Profit Margin %', ascending=False)
    st.dataframe(
        subcat_display.style.format({
            'Sales': '${:,.0f}',
            'Profit': '${:,.0f}',
            'Profit Margin %': '{:.2f}%'
        }).background_gradient(subset=['Profit Margin %'], cmap='RdYlGn'),
        use_container_width=True
    )


# ===== TAB 5: REGIONAL ANALYSIS =====
with tab5:
    st.header("🗺️ Regional Performance")
    region_data = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Customer ID': 'nunique'
    }).reset_index()
    region_data['Profit Margin %'] = round(
        region_data['Profit'] / region_data['Sales'] * 100, 2)
    region_data['Avg Order Value'] = region_data['Sales'] / \
        region_data['Order ID']
    region_data = region_data.sort_values(by="Sales", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(region_data, x='Region', y='Sales',
                     color='Profit Margin %', color_continuous_scale='RdYlGn',
                     title='Regional Sales Performance', text='Sales')
        fig.update_traces(
            texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(template='plotly_white', height=400)
        fig.update_yaxes(
            range=[0, 1000000],
            tickvals=[0, 200000, 400000, 600000, 800000, 1000000],
            ticktext=['0', '200K', '400K', '600K', '800K', '1,000K']
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(region_data, values='Profit', names='Region',
                     title='Profit Distribution by Region', hole=0.4)
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Regional metrics
    st.subheader("Regional Metrics Comparison")
    st.dataframe(
        region_data.sort_values(by='Profit Margin %', ascending=False).style.format({
            'Sales': '${:,.0f}',
            'Profit': '${:,.0f}',
            'Profit Margin %': '{:.2f}%',
            'Avg Order Value': '${:.2f}'
        }).background_gradient(subset=['Profit Margin %'], cmap='RdYlGn'),
        use_container_width=True
    )


# ===== TAB 6: KEY FINDINGS =====
with tab6:
    st.header("🎯 Key Business Insights & Recommendations")

    st.markdown("""
    <div class="insight-box">
    <h3 style = 'color: #444'>🚨 Critical Finding #1: The Profit Cliff</h3>
    <p><strong style = 'color: #444'>Issue:</strong> 96.77% of orders with 25%+ discounts are unprofitable</p>
    <p><strong style = 'color: #444'>Impact:</strong> Estimated annual loss of $39,032.82 from excessive discounting</p>
    <p><strong style = 'color: #444'>Recommendation:</strong> Implement hard cap at 20% discount with VP approval required for exceptions</p>
    <p><strong style = 'color: #444'>Projected Benefit:</strong> $458,307.92 annual profit recovery</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h3 style = 'color: #444'>💎 Critical Finding #2: Customer Lifecycle Volatility</h3>
    <p><strong  style = 'color: #444'>Issue:</strong> Orders 5, 7, 10, 14 show 40% profit inconsistency despite 16% average discounts</p>
    <p ><strong  style = 'color: #444'>Root Cause:</strong> Blanket loyalty discounts without strategic targeting</p>
    <p><strong  style = 'color: #444'>Recommendation:</strong> Replace % discounts with category-based rewards and tiered incentives</p>
    <p><strong  style = 'color: #444'>Projected Benefit:</strong> $35,000 annual profit increase</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h3 style = 'color: #444'>🎯 Critical Finding #3: First Purchase Matters</h3>
    <p><strong  style = 'color: #444'>Finding:</strong> Furniture-first customers show 25.3% lower lifetime value vs Technology-first customers</p>
    <p><strong  style = 'color: #444'>Recommendation:</strong> Shift new customer acquisition incentives toward Technology category</p>
    <p><strong  style = 'color: #444'>Projected Benefit:</strong> Improved customer acquisition ROI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h3  style = 'color: #444'>💰 Critical Finding #4: Strategic Loss Leaders</h3>
    <p><strong  style = 'color: #444'>Finding:</strong> Some "unprofitable" products drive 3x future profitable purchases</p>
    <p><strong  style = 'color: #444'>Recommendation:</strong> Maintain these products as relationship builders, not eliminate them</p>
    <p><strong  style = 'color: #444'>Projected Benefit:</strong> Preserve $14215.32 in future revenue streams</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h3  style = 'color: #444'>🎄 Critical Finding #5: Q4 Profitability Paradox</h3>
    <p><strong  style = 'color: #444'>Finding:</strong> Q4 generates highest revenue but lowest profit margins due to aggressive discounting</p>
    <p><strong  style = 'color: #444'>Recommendation:</strong> Reduce Q4 discount rates by 5% (from 19% to 14%)</p>
    <p><strong  style = 'color: #444'>Projected Benefit:</strong> $4091.94 additional Q4 profit without losing volume</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("""
📊 Total Annual Opportunity: $$458,000+""")

    st.success("""
    **Implementation Priority:**
    1. **Immediate (Week 1):** Halt 25%+ discounts, block in system
    2. **Month 1:** Implement tiered loyalty program
    3. **Month 2:** Adjust acquisition strategy toward Technology category
    4. **Month 3:** Q4 discount policy revision
    5. **Ongoing:** Monitor loss leader performance quarterly
    """)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>📊 Sales Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>👨‍💻 Developed by Adeel Manaf | Data Scientist</p>
        <p>🔗 <a href='https://github.com/adeelmanaf43'>GitHub</a> | 
           <a href='https://linkedin.com/in/adeel-manaf-0b1a70194'>LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)
