# edm_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# import the robust loader and model from your loader module
# make sure edm_data_loader.py is in the same folder or in PYTHONPATH
from edm_data_loader import load_edm_dataset, EDMModel

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a8c 50%, #1e3a5f 100%);
        border-right: 2px solid #ff6b6b;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
        font-weight: 500;
    }
    .main-title { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4444 100%); 
        padding: 2px; 
        border-radius: 15px; 
        margin-bottom: 20px; 
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.6), 0 0 60px rgba(255, 68, 68, 0.3); 
    }
    .main-title-content { 
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); 
        padding: 25px; 
        border-radius: 13px; 
        text-align: center; 
    }
    .main-title-content h1 { 
        color: #ff6b6b; 
        margin: 0; 
        text-shadow: 0 0 15px rgba(255, 107, 107, 0.8); 
        font-size: 2.5em; 
        font-weight: 900; 
        letter-spacing: 2px; 
    }
    .section-header { 
        background: linear-gradient(90deg, #cc0000 0%, #ff6b6b 100%); 
        padding: 2px; 
        border-radius: 10px; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        box-shadow: 0 0 20px rgba(204, 0, 0, 0.5), 0 0 40px rgba(255, 107, 107, 0.3); 
    }
    .section-header h2 { 
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); 
        padding: 15px; 
        margin: 0; 
        border-radius: 8px; 
        color: #ff8888; 
        text-shadow: 0 0 10px rgba(255, 136, 136, 0.7); 
        font-size: 1.6em; 
        font-weight: 800; 
    }
    .stMarkdownContainer, p, span { 
        color: #ffffff; 
        font-weight: 500; 
    }
    [data-testid="metric-container"] { 
        background: linear-gradient(135deg, #4f2a2a 0%, #8c2d2d 100%); 
        border: 2px solid #ff6b6b; 
        border-radius: 12px; 
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.4); 
        padding: 20px; 
    }
    [data-testid="metric-container"] label { 
        color: #ffaaaa; 
        font-weight: bold; 
        text-shadow: 0 0 8px rgba(255, 170, 170, 0.5); 
    }
    [data-testid="metric-container"] [data-testid="metric-value"] { 
        color: #ff6b6b; 
        font-size: 2em; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.6); 
    }
    .stSuccess { 
        background-color: rgba(204, 0, 0, 0.2) !important; 
        border: 2px solid #ff6b6b !important; 
        border-radius: 10px !important; 
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.3) !important; 
    }
    .stInfo { 
        background-color: rgba(255, 107, 107, 0.15) !important; 
        border: 2px solid #ff8888 !important; 
        border-radius: 10px !important; 
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.3) !important; 
    }
    .stWarning { 
        background-color: rgba(255, 153, 0, 0.15) !important; 
        border: 2px solid #ff9900 !important; 
        border-radius: 10px !important; 
    }
    .stSelectbox label, .stSlider label, .stRadio label { 
        color: #ffffff !important; 
        font-weight: 600 !important; 
        text-shadow: 0 0 5px rgba(255, 107, 107, 0.4) !important; 
    }
    hr { 
        border: 1px solid rgba(255, 107, 107, 0.3) !important; 
    }
    [data-testid="stDataframe"] { 
        background: linear-gradient(135deg, #4f2a2a 0%, #8c2d2d 100%) !important; 
        border: 1px solid #ff6b6b !important; 
        border-radius: 10px !important; 
    }
    .stCaption { 
        color: #ff8888 !important; 
        text-shadow: 0 0 5px rgba(255, 136, 136, 0.4) !important; 
    }
    [data-testid="stSidebar"] h2 { 
        color: #ffaaaa !important; 
        text-shadow: 0 0 8px rgba(255, 170, 170, 0.6) !important; 
        border-bottom: 2px solid #ff6b6b !important; 
        padding-bottom: 10px !important; 
    }
    .stRadio > label { 
        color: #ffffff !important; 
    }
    .stRadio > label > span:first-child { 
        background-color: transparent !important; 
        border: 2px solid #ff6b6b !important; 
        border-radius: 6px !important; 
    }
    .stSlider [role="slider"] { 
        background: linear-gradient(90deg, #cc0000, #ff6b6b) !important; 
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Employment Displacement Model (EDM)",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Main Title with Glowing Effect
# ---------------------------
st.markdown("""
    <div class="main-title">
        <div class="main-title-content">
            <h1>⚠️ Employment Displacement Model – EDM</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(200, 20, 20, 0.08)); border: 2px solid #ff6b6b; border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(255, 107, 107, 0.25);">
    <p style="color: #ffffff; font-size: 1.05em; line-height: 1.8; margin: 0; font-weight: 500;">
        The <b>Employment Displacement Model (EDM)</b> is a comprehensive analytical framework designed to quantify and visualize the employment vulnerability of nations in an increasingly automated world. By integrating real-world data from the International Labour Organization (ILOSTAT) with advanced mathematical models, our platform reveals how automation speed, wage dynamics, and skill investment collectively influence employment stability across different countries and sectors.
    </p>
</div>

<p style="text-align: center; color: #ff8888; font-size: 1.08em; margin-top: 15px; margin-bottom: 25px; text-shadow: 0 0 8px rgba(255, 136, 136, 0.5); font-weight: 600;">
Welcome to the Humanoid Work Math Project — explore how automation shapes employment displacement and job risk worldwide.
</p>

st.divider()
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio("Select Mode:", ["Manual Simulation", "ILOSTAT Dataset"])

# Manual Simulation
if mode == "Manual Simulation":
    st.markdown('<div class="section-header"><h2>🧮 EDM Simulation Mode</h2></div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#e0e0e0; font-size:1.03em; margin-bottom:12px;'>Adjust automation speed (A), baseline employment (D₀), sensitivity (β), and time horizon (t) to see expected displacement outcomes.</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        A = st.slider("Automation Speed (A)", 0.0, 1.0, 0.4, step=0.05)
    with col2:
        employment_input = st.number_input("Baseline Employment (D₀)", min_value=100.0, max_value=10000000.0, value=100000.0, step=1000.0, format="%.0f")
    with col3:
        beta = st.slider("Sensitivity (β)", 0.01, 1.0, 0.30, step=0.01)
    with col4:
        time_years = st.slider("Time Horizon (t, years)", 0, 30, 10, step=1)

    st.info(f"Model: D(t) = D₀ × e^(βAt)  –  β = {beta:.2f}, t = {time_years} years")

    model = EDMModel(beta=beta)
    D = model.compute_edm_raw(employment_input, A, time_years)
    EDM_pct = model.compute_edm_percent(employment_input, A, time_years)
    EDM_index_manual = model.compute_edm_index(employment_input, A, time_years)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("⚠️ Displaced Jobs", f"{D:,.0f}")
    with metric_col2:
        st.metric("📊 Displacement Increase (%)", f"{EDM_pct*100:.2f}%")
    with metric_col3:
        st.metric("📈 EDM (index 0-1)", f"{EDM_index_manual:.3f}")

    st.markdown('<div class="section-header"><h2>📈 EDM Index vs Automation Speed</h2></div>', unsafe_allow_html=True)
    A_values = np.linspace(0, 1, 50)
    D_values = [model.compute_edm_raw(employment_input, a, time_years) for a in A_values]
    EDM_pct_values = [model.compute_edm_percent(employment_input, a, time_years) for a in A_values]
    EDM_index_values = [model.compute_edm_index(employment_input, a, time_years) for a in A_values]

    fig_manual = go.Figure()
    fig_manual.add_trace(go.Scatter(
        x=A_values,
        y=EDM_index_values,
        mode="lines+markers",
        line=dict(color="#ff6b6b", width=4),
        marker=dict(size=8, color="#ff8888", line=dict(width=1, color="#ffffff")),
        hovertemplate="A: %{x:.2f}<br>EDM index: %{y:.3f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.08)"
    ))
    fig_manual.update_layout(
        xaxis=dict(title=dict(text="Automation Speed (A)", font=dict(color="#ff6b6b")), tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title=dict(text="EDM Index (0–1)", font=dict(color="#ff6b6b")), tickfont=dict(color="#e0e0e0"), range=[0, 1]),
        template="plotly_dark",
        plot_bgcolor="rgba(15,20,25,0.8)",
        paper_bgcolor="rgba(26,31,46,0.9)",
        height=450,
        font=dict(color="#e0e0e0"),
        margin=dict(l=70, r=30, t=30, b=70)
    )
    st.plotly_chart(fig_manual, use_container_width=True)
    st.divider()

    st.markdown("""
    <div class="section-header"><h2>🧮 Mathematical Formula & Methodology</h2></div>

    <p style="color:#e0e0e0; font-size:1.03em; margin-bottom:18px;">
    The <b>Employment Displacement Model (EDM)</b> estimates how automation displaces jobs over time.
    We use baseline employment as D₀ and model expected displacement as:
    </p>
    """, unsafe_allow_html=True)

    formula_col1, formula_col2 = st.columns(2)
    with formula_col1:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="color:#ff99ff; font-weight:800; margin-bottom:15px;">Base Formula:</h3>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"D(t) = D_0 \times e^{\beta A t}")
        st.markdown("""
        <div style="margin-top:20px;">
            <p style="color:#ffffff; font-size:16px; font-weight:600; margin-bottom:12px;">Where:</p>
            <ul style="color:#cbd5e1; font-size:14px; line-height:2.0; list-style-position: inside;">
                <li><b style="color:#ff99ff;">A</b> (Automation Speed) - Rate of technological displacement (0-1)</li>
                <li><b style="color:#ff99ff;">D₀</b> (Baseline Employment) - Initial jobs at risk</li>
                <li><b style="color:#ff99ff;">β</b> (Sensitivity) - Displacement sensitivity to automation</li>
                <li><b style="color:#ff99ff;">t</b> (Time) - Years horizon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with formula_col2:
        st.markdown('<div style="margin-bottom: 20px;"><h3 style="color:#ff66ff; font-weight:800; margin-bottom:15px;">Derived Metrics:</h3></div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:40px;"><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">Percent Displacement:</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7; margin-bottom:20px;">ΔD% = (D(t) - D₀) / D₀</p></div>', unsafe_allow_html=True)
        st.markdown('<div><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">Time to Threshold:</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7; margin-bottom:20px;">t = ln(1+ΔD%) / (βA)</p></div>', unsafe_allow_html=True)
        st.markdown('<div><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">EDM Index (0-1):</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7;">Logistic scaling of displacement percentage</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:0; height:1px; background:rgba(255,107,107,0.12); margin:30px 0;">
    <div style="background: linear-gradient(135deg, rgba(255, 107, 107, 0.12), rgba(200, 20, 20, 0.18)); border-left:4px solid #ff0000; padding:20px; border-radius:8px;">
      <h3 style="color:#ff9999; margin-top:0; margin-bottom:15px;">📘 Interpretation Guide:</h3>
      <ul style="color:#f0f3ff; line-height:2.0; font-size:14px;">
        <li><b style="color:#00ff00;">● EDM 0.0 - 0.33 (🟢 Low Risk):</b> Minimal displacement; stable employment with gradual automation</li>
        <li><b style="color:#ffaa00;">● EDM 0.33 - 0.67 (🟡 Moderate Risk):</b> Significant job displacement requiring active retraining programs</li>
        <li><b style="color:#ff0000;">● EDM 0.67 - 1.0 (🔴 High Risk):</b> Severe displacement risk; urgent workforce intervention needed</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ILOSTAT Dataset mode
elif mode == "ILOSTAT Dataset":
    st.markdown('<div class="section-header"><h2>🌍 ILOSTAT EDM Integration</h2></div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #e0e0e0; font-size:1.03em; margin-bottom:12px;'>EDM computed from ILOSTAT (latest year per country). Employment derived from employment-to-population ratio; A computed consistently with the ERI pipeline.</p>", unsafe_allow_html=True)

    data = load_edm_dataset()

    if data.empty:
        st.error("⚠️ Could not load EDM dataset. Please check your data files.")
    else:
        # latest per country (safely handle Year missing)
        if "Year" in data.columns:
            latest = data.sort_values("Year").drop_duplicates("Area", keep="last")
        else:
            latest = data.copy()

        st.markdown('<div class="section-header"><h2>🌍 Global Employment Displacement Index (Latest Year)</h2></div>', unsafe_allow_html=True)

        # try choropleth map
        try:
            fig_map = px.choropleth(
                latest,
                locations="Area",
                locationmode="country names",
                color="EDM_index",
                color_continuous_scale="Reds",
                range_color=(0, 1),
                hover_name="Area",
                hover_data={
                    "Year": True,
                    "EDM_raw": ":,.0f",
                    "EDM_pct": ":.3f",
                    "A": ":.2f",
                    "Employment": ":,.0f"
                },
                title="Employment Displacement Index (EDM) – Relative (Latest Year)"
            )
            fig_map.update_geos(
                showcountries=True,
                countrycolor="rgba(255,255,255,0.15)")
            fig_map.update_traces(marker_line_width=0.4, marker_line_color="white")
            fig_map.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(15, 20, 25, 0.6)"),
                margin=dict(l=0, r=0, t=50, b=0),
                height=560,
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                font=dict(color="#e0e0e0"),
                coloraxis_colorbar=dict(title="EDM (index 0–1)")
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning("Map rendering failed (likely due to country name mismatches). Showing table and charts only. Error: " + str(e))

        st.divider()

        # Top & Bottom countries
        st.markdown('<div class="section-header"><h2>📊 Country Comparison – Highest vs Lowest EDM</h2></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="color:#ff6b6b; font-weight:bold; text-align:center;">🔴 Highest EDM Countries</p>', unsafe_allow_html=True)
            top = latest.sort_values(by="EDM_index", ascending=False).head(10)
            fig_top = go.Figure(go.Bar(
                x=top["EDM_index"],
                y=top["Area"],
                orientation="h",
                marker=dict(color=top["EDM_index"], colorscale="Reds", line=dict(width=1, color="#ffffff")),
                text=[f"{(p*100):.2f}%" for p in top["EDM_pct"].fillna(0)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>EDM pct: %{text}<br>Index: %{x:.3f}<extra></extra>"
            ))
            fig_top.update_layout(
                height=420,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="EDM Index", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15,20,25,0.8)",
                paper_bgcolor="rgba(26,31,46,0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            st.markdown('<p style="color:#90caf9; font-weight:bold; text-align:center;">🟢 Lowest EDM Countries</p>', unsafe_allow_html=True)
            low = latest.sort_values(by="EDM_index", ascending=True).head(10)
            fig_low = go.Figure(go.Bar(
                x=low["EDM_index"],
                y=low["Area"],
                orientation="h",
                marker=dict(color=low["EDM_index"], colorscale="Reds_r", line=dict(width=1, color="#ffffff")),
                text=[f"{(p*100):.2f}%" for p in low["EDM_pct"].fillna(0)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>EDM pct: %{text}<br>Index: %{x:.3f}<extra></extra>"
            ))
            fig_low.update_layout(
                height=420,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="EDM Index", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15,20,25,0.8)",
                paper_bgcolor="rgba(26,31,46,0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_low, use_container_width=True)

        st.divider()

        # Scatter A vs EDM_index
        st.markdown('<div class="section-header"><h2>⚖️ Automation (A) vs EDM (index)</h2></div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            latest,
            x="A",
            y="EDM_index",
            color="EDM_index",
            color_continuous_scale="Reds",
            range_color=(0, 1),
            hover_name="Area",
            hover_data={"EDM_pct": ":.3f", "Employment": ":,.0f"},
            labels={"A": "Automation Speed (A)", "EDM_index": "EDM (index 0–1)"},
            title="Automation Speed vs Employment Displacement (relative index)"
        )
        fig_scatter.update_yaxes(range=[0, 1])
        fig_scatter.update_layout(
            plot_bgcolor="rgba(15,20,25,0.8)",
            paper_bgcolor="rgba(26,31,46,0.9)",
            font=dict(color="#e0e0e0"),
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.divider()

        # Full table
        st.markdown('<div class="section-header"><h2>🧾 Full EDM Data Table</h2></div>', unsafe_allow_html=True)
        st.dataframe(
            latest[["Area", "Year", "A", "Employment", "Population", "EmpPop", "Unemp", "TimeYears", "EDM_raw", "EDM_pct", "EDM_index"]].sort_values(by="EDM_index", ascending=False),
            use_container_width=True,
            height=450
        )

# Footer
st.divider()
st.markdown("""
<p style="text-align: center; color: #ff8888; font-size: 0.9em; margin-top: 20px; text-shadow: 0 0 8px rgba(255, 107, 107, 0.4);">
Built with ❤️ by the <b>Humanoid Robots and Future of Work Team (CUT)</b> – Employment Displacement Index (EDM) analysis.
</p>
""", unsafe_allow_html=True)