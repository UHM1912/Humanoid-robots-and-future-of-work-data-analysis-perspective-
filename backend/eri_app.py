import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from eri_model import ERIModel
from eri_data_loader import load_ilostat_data

# ---------------------------
# Custom CSS for Cyan/Teal Theme with Glowing Effects
# ---------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3a42 0%, #1a5a6f 50%, #0d3a42 100%);
        border-right: 2px solid #00d9d9;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Main Title with Glowing Border */
    .main-title {
        background: linear-gradient(135deg, #00d9d9 0%, #00a3a3 100%);
        padding: 2px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0, 217, 217, 0.6), 
                    0 0 60px rgba(0, 163, 163, 0.3);
    }
    
    .main-title-content {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 25px;
        border-radius: 13px;
        text-align: center;
    }
    
    .main-title-content h1 {
        color: #00d9d9;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 217, 217, 0.8);
        font-size: 2.5em;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    /* Section Headers with Glowing Effect */
    .section-header {
        background: linear-gradient(90deg, #008080 0%, #00d9d9 100%);
        padding: 2px;
        border-radius: 10px;
        margin-top: 25px;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(0, 128, 128, 0.5),
                    0 0 40px rgba(0, 217, 217, 0.3);
    }
    
    .section-header h2 {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 15px;
        margin: 0;
        border-radius: 8px;
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
        font-size: 1.6em;
        font-weight: 800;
    }
    
    /* Text Color */
    .stMarkdownContainer, p, span {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d3a42 0%, #1a5a6f 100%);
        border: 2px solid #00d9d9;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 217, 217, 0.4);
        padding: 20px;
    }
    
    [data-testid="metric-container"] label {
        color: #00ffff;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #00d9d9;
        font-size: 2em;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 217, 217, 0.6);
    }
    
    /* Success/Info Boxes */
    .stSuccess {
        background-color: rgba(0, 128, 128, 0.2) !important;
        border: 2px solid #00d9d9 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 217, 217, 0.3) !important;
    }
    
    .stInfo {
        background-color: rgba(0, 217, 217, 0.15) !important;
        border: 2px solid #00a3a3 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 217, 217, 0.3) !important;
    }
    
    .stWarning {
        background-color: rgba(255, 153, 0, 0.15) !important;
        border: 2px solid #ff9900 !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar Labels */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0 0 5px rgba(0, 217, 217, 0.4) !important;
    }
    
    /* Divider */
    hr {
        border: 1px solid rgba(0, 217, 217, 0.3) !important;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataframe"] {
        background: linear-gradient(135deg, #0d3a42 0%, #1a5a6f 100%) !important;
        border: 1px solid #00d9d9 !important;
        border-radius: 10px !important;
    }
    
    /* Caption */
    .stCaption {
        color: #00d9d9 !important;
        text-shadow: 0 0 5px rgba(0, 217, 217, 0.4) !important;
    }
    
    /* Latex */
    .stLatex {
        color: #00d9d9 !important;
    }
    
    /* Sidebar Header */
    [data-testid="stSidebar"] h2 {
        color: #00ffff !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6) !important;
        border-bottom: 2px solid #00d9d9 !important;
        padding-bottom: 10px !important;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #ffffff !important;
    }
    
    .stRadio > label > span:first-child {
        background-color: transparent !important;
        border: 2px solid #00d9d9 !important;
        border-radius: 6px !important;
    }
    
    .stRadio > label > span:first-child > svg {
        fill: #00ffff !important;
    }
    
    /* Slider */
    .stSlider [role="slider"] {
        background: linear-gradient(90deg, #00a3a3, #00d9d9) !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Employment Risk Index (ERI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Main Title with Glowing Effect and Paper-Style Intro
# ---------------------------
st.markdown("""
    <div class="main-title">
        <div class="main-title-content">
            <h1>🤖 Employment Risk Index – ERI</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(0, 217, 217, 0.15), rgba(0, 163, 163, 0.08)); border: 2px solid #00d9d9; border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 217, 217, 0.25);">
    <p style="color: #ffffff; font-size: 1.05em; line-height: 1.8; margin: 0; font-weight: 500;">
        The <b>Employment Risk Index (ERI)</b> is a comprehensive analytical framework designed to quantify and visualize the employment vulnerability of nations in an increasingly automated world. By integrating real-world data from the International Labour Organization (ILOSTAT) with advanced mathematical models, our platform reveals how automation speed, wage dynamics, and skill investment collectively influence employment stability across different countries and sectors.
    </p>
</div>

<p style="text-align: center; color: #00d9d9; font-size: 1.08em; margin-top: 15px; margin-bottom: 25px; text-shadow: 0 0 8px rgba(0, 217, 217, 0.5); font-weight: 600;">
Welcome to the Humanoid Work Math Project — explore how automation, wages, and skills shape employment risk worldwide.
</p>

st.divider()
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio("Select Mode:", ["Manual Simulation", "ILOSTAT Dataset"])

# ==============================================================
# 1️⃣ MANUAL SIMULATION MODE
# ==============================================================

if mode == "Manual Simulation":
    st.markdown("""
        <div class="section-header">
            <h2>🧮 ERI Simulation Mode</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #e0e0e0; font-size: 1.05em; margin-bottom: 20px;">
    Experiment with automation speed (<b>A</b>), wage factor (<b>W</b>), and skill investment (<b>S</b>) 
    to observe how they influence the Employment Risk Index.
    </p>
    """, unsafe_allow_html=True)

    # Input sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        A = st.slider("Automation Speed (A)", 0.0, 1.0, 0.4, step=0.05)
    with col2:
        W = st.slider("Wage Factor (W)", 0.5, 2.0, 1.0, step=0.1)
    with col3:
        S = st.slider("Skill Investment (S)", 0.0, 5.0, 2.0, step=0.1)

    # Model type
    model_type = st.selectbox(
        "Select ERI Model Type:",
        ["linear", "quadratic", "exponential", "logistic"]
    )

    explanations = {
        "linear": "📈 **Linear:** Risk rises steadily with automation.",
        "quadratic": "📉 **Quadratic:** Risk accelerates faster at high automation.",
        "exponential": "🚀 **Exponential:** Risk surges sharply after a threshold.",
        "logistic": "⚖️ **Logistic:** S-shaped curve — risk spikes then stabilizes."
    }
    st.info(explanations[model_type])

    # Model computation
    model = ERIModel()
    eri_value = model.compute_eri(A, W, S, mode=model_type)
    risk_level = model.interpret(eri_value)

    # Display metrics in columns
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("🎯 Employment Risk Index", f"{eri_value:.3f}", delta=None)
    with metric_col2:
        st.metric("⚠️ Risk Level", risk_level, delta=None)
    with metric_col3:
        status_emoji = "🟢" if eri_value < 0.33 else "🟡" if eri_value < 0.67 else "🔴"
        st.metric("📊 Status", status_emoji, delta=None)

    # Generate range for visualization
    A_values = np.linspace(0, 1, 50)
    df = pd.DataFrame({
        "Automation Speed": A_values,
        "ERI": [model.compute_eri(a, W, S, mode=model_type) for a in A_values]
    })
    df["Risk"] = df["ERI"].apply(model.interpret)

    # Plotly graph with risk zones
    st.markdown("""
        <div class="section-header">
            <h2>📈 ERI vs Automation Speed</h2>
        </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Automation Speed"],
        y=df["ERI"],
        mode="lines+markers",
        line=dict(color="#00d9d9", width=4),
        marker=dict(size=10, color=df["ERI"], colorscale="RdYlGn_r", line=dict(width=2, color="#ffffff")),
        text=[f"ERI: {v:.3f}<br>Risk: {r}" for v, r in zip(df["ERI"], df["Risk"])],
        hoverinfo="text",
        fill="tozeroy",
        fillcolor="rgba(0, 217, 217, 0.1)"
    ))
    
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=0.2,
                  fillcolor="green", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.2, y1=0.6,
                  fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.6, y1=1.0,
                  fillcolor="red", opacity=0.2, layer="below", line_width=0)
    
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Automation Speed (A)", font=dict(color="#00d9d9", size=14)),
            tickfont=dict(color="#e0e0e0"),
            gridcolor="rgba(0, 217, 217, 0.1)",
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="Employment Risk Index (ERI)", font=dict(color="#00d9d9", size=14)),
            tickfont=dict(color="#e0e0e0"),
            gridcolor="rgba(0, 217, 217, 0.1)",
            showgrid=True
        ),
        yaxis_range=[0, 1],
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        plot_bgcolor="rgba(15, 20, 25, 0.8)",
        paper_bgcolor="rgba(26, 31, 46, 0.9)",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=70, r=30, t=30, b=70)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    st.markdown("""
    <p style="color: #00ffff; font-weight: bold; font-size: 1.1em; text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);">Formula:</p>
    """, unsafe_allow_html=True)
    
    st.latex(r"ERI = \frac{A \times W}{S + 1}")
    
    st.markdown("""
    <div class="section-header">
        <h2>📐 Mathematical Formula & Methodology</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #ffffff; font-size: 1.05em; margin-bottom: 20px; line-height: 1.7;">
    The <b>Employment Risk Index (ERI)</b> combines three critical variables to assess employment vulnerability:
    </p>
    """, unsafe_allow_html=True)
    
    col_formula1, col_formula2 = st.columns([1, 1])
    
    with col_formula1:
        st.markdown("""
        <p style="color: #00ffff; font-weight: bold; font-size: 1.1em; margin-bottom: 15px;">
        Base Formula:
        </p>
        """, unsafe_allow_html=True)
        st.latex(r"ERI = \frac{A \times W}{S + 1}")
        
        st.markdown("""
        <p style="color: #ffffff; font-size: 0.95em; margin-top: 20px; line-height: 1.8;">
        <b>Where:</b><br>
        <span style="color: #00ffff;">• A (Automation Speed)</span> - Rate of technological displacement (0-1)<br>
        <span style="color: #00ffff;">• W (Wage Factor)</span> - Economic pressure from wage levels (0.5-2.0)<br>
        <span style="color: #00ffff;">• S (Skill Investment)</span> - Human capital development (0-5)
        </p>
        """, unsafe_allow_html=True)
    
    with col_formula2:
        st.markdown("""
        <p style="color: #00ffff; font-weight: bold; font-size: 1.1em; margin-bottom: 15px;">
        Non-linear Models:
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="color: #ffffff; font-size: 0.95em; line-height: 1.8;">
        <b>Quadratic Model:</b><br>
        <span style="color: #cccccc;">Emphasizes acceleration of risk at higher automation levels</span><br><br>
        
        <b>Exponential Model:</b><br>
        <span style="color: #cccccc;">Models rapid risk escalation beyond critical thresholds</span><br><br>
        
        <b>Logistic Model:</b><br>
        <span style="color: #cccccc;">S-shaped curve representing risk saturation and market adaptation</span>
        </p>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <p style="color: #cccccc; font-size: 0.95em; background: rgba(0, 128, 128, 0.1); padding: 15px; border-left: 4px solid #00d9d9; border-radius: 5px; line-height: 1.8;">
    <b>📊 Interpretation Guide:</b><br>
    • <b>ERI 0.0 - 0.33 (🟢 Low Risk):</b> Strong employment stability with balanced automation and skills<br>
    • <b>ERI 0.33 - 0.67 (🟡 Moderate Risk):</b> Emerging challenges requiring skill investment and policy intervention<br>
    • <b>ERI 0.67 - 1.0 (🔴 High Risk):</b> Significant employment vulnerability demanding urgent workforce development
    </p>
    """, unsafe_allow_html=True)

# ==============================================================
# 2️⃣ ILOSTAT DATASET MODE
# ==============================================================

elif mode == "ILOSTAT Dataset":
    st.markdown("""
        <div class="section-header">
            <h2>🌍 ILOSTAT Data Integration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #e0e0e0; font-size: 1.05em; margin-bottom: 20px;">
    View the <b>Employment Risk Index (ERI)</b> computed from real-world ILOSTAT indicators across countries.
    Each country represents its <b>latest available year</b> of data.
    </p>
    """, unsafe_allow_html=True)

    # Load dataset
    data = load_ilostat_data()
    data = data.dropna(subset=["ERI"])

    if data.empty:
        st.warning("⚠️ No ERI data available. Ensure all ILOSTAT CSVs are in the same folder.")
    else:
        # Get latest year per country
        latest = data.sort_values("Year").drop_duplicates("Area", keep="last")

        # ---------------------------
        # 🌍 Global ERI Map
        # ---------------------------
        st.markdown("""
            <div class="section-header">
                <h2>🌎 Global Employment Risk Map</h2>
            </div>
        """, unsafe_allow_html=True)

        fig_map = px.choropleth(
            latest,
            locations="Area",
            locationmode="country names",
            color="ERI",
            color_continuous_scale="RdYlGn_r",
            range_color=(0, 1),
            hover_name="Area",
            hover_data={
                "Year": True,
                "ERI": ":.3f",
                "A": ":.2f",
                "W": ":.2f",
                "S": ":.2f"
            },
            title="Global Employment Risk Index (Latest Year)"
        )

        fig_map.update_traces(
            marker_line_color="white",
            marker_line_width=0.5
        )

        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(15, 20, 25, 0.6)"),
            coloraxis_colorbar=dict(
                title="ERI",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["Low", "→", "Moderate", "→", "High", "Very High"],
                thickness=20,
                len=0.7
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=550,
            paper_bgcolor="rgba(26, 31, 46, 0.9)",
            plot_bgcolor="rgba(15, 20, 25, 0.8)",
            font=dict(color="#e0e0e0", size=11),
            title=dict(font=dict(color="#00d9d9", size=18))
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.divider()

        # ---------------------------
        # 📊 Top & Bottom 10 Countries
        # ---------------------------
        st.markdown("""
            <div class="section-header">
                <h2>📈 Country Comparison — Highest vs Lowest ERI</h2>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <p style="color: #ff6b6b; font-weight: bold; font-size: 1.1em; text-align: center; text-shadow: 0 0 8px rgba(255, 107, 107, 0.5);">
            🔺 Highest ERI Countries
            </p>
            """, unsafe_allow_html=True)
            
            top = latest.sort_values(by="ERI", ascending=False).head(10)
            fig_top = go.Figure(go.Bar(
                x=top["ERI"],
                y=top["Area"],
                orientation="h",
                marker=dict(color=top["ERI"], colorscale="Teal", line=dict(width=2, color="#ffffff")),
                text=[f"{v:.3f}" for v in top["ERI"]],
                textposition="outside",
                textfont=dict(color="#ffffff", size=11),
                hovertemplate="<b>%{y}</b><br>ERI: %{x:.3f}<extra></extra>"
            ))
            fig_top.update_layout(
                height=450,
                xaxis=dict(
                    title=dict(text="Employment Risk Index", font=dict(color="#00d9d9", size=12)),
                    tickfont=dict(color="#e0e0e0")
                ),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                margin=dict(l=120, r=80, t=20, b=40),
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0", size=11),
                hovermode="closest"
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            st.markdown("""
            <p style="color: #51cf66; font-weight: bold; font-size: 1.1em; text-align: center; text-shadow: 0 0 8px rgba(81, 207, 102, 0.5);">
            🟢 Lowest ERI Countries
            </p>
            """, unsafe_allow_html=True)
            
            low = latest.sort_values(by="ERI", ascending=True).head(10)
            fig_low = go.Figure(go.Bar(
                x=low["ERI"],
                y=low["Area"],
                orientation="h",
                marker=dict(color=low["ERI"], colorscale="Greens", line=dict(width=2, color="#ffffff")),
                text=[f"{v:.3f}" for v in low["ERI"]],
                textposition="outside",
                textfont=dict(color="#ffffff", size=11),
                hovertemplate="<b>%{y}</b><br>ERI: %{x:.3f}<extra></extra>"
            ))
            fig_low.update_layout(
                height=450,
                xaxis=dict(
                    title=dict(text="Employment Risk Index", font=dict(color="#00d9d9", size=12)),
                    tickfont=dict(color="#e0e0e0")
                ),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                margin=dict(l=120, r=80, t=20, b=40),
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0", size=11),
                hovermode="closest"
            )
            st.plotly_chart(fig_low, use_container_width=True)

        st.divider()
        
        st.markdown("""
            <div class="section-header">
                <h2>🧾 Full ERI Data Table</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="color: #e0e0e0; font-size: 0.95em; margin-bottom: 15px;">
        <i>Latest Year Data by Country</i>
        </p>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            latest[["Area", "Year", "A", "W", "S", "ERI"]].sort_values(by="ERI", ascending=False),
            use_container_width=True,
            height=500
        )

# ==============================================================
# Footer
# ==============================================================
st.divider()
st.markdown("""
<p style="text-align: center; color: #00d9d9; font-size: 0.9em; margin-top: 20px; text-shadow: 0 0 8px rgba(0, 217, 217, 0.4);">
Built with ❤️ by the <b>Humanoid Robots and Future of Work Team (CUT)</b> — Analyzing automation impacts on global employment.
</p>
""", unsafe_allow_html=True)