
# pgi_app.py (updated with paper style)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# import the robust loader and model from your loader module
# make sure pgi_data_loader.py is in the same folder or in PYTHONPATH
from pgi_data_loader import load_pgi_dataset, PGIModel

# ---------------------------
# Custom CSS (same as you had)
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
        border-right: 2px solid #00d4ff;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
        font-weight: 500;
    }
    .main-title { 
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); 
        padding: 2px; 
        border-radius: 15px; 
        margin-bottom: 20px; 
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6), 0 0 60px rgba(0, 153, 204, 0.3); 
    }
    .main-title-content { 
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); 
        padding: 25px; 
        border-radius: 13px; 
        text-align: center; 
    }
    .main-title-content h1 { 
        color: #00d4ff; 
        margin: 0; 
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.8); 
        font-size: 2.5em; 
        font-weight: 900; 
        letter-spacing: 2px; 
    }
    .section-header { 
        background: linear-gradient(90deg, #0066cc 0%, #00ccff 100%); 
        padding: 2px; 
        border-radius: 10px; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        box-shadow: 0 0 20px rgba(0, 102, 204, 0.5), 0 0 40px rgba(0, 204, 255, 0.3); 
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
    .stMarkdownContainer, p, span { 
        color: #ffffff; 
        font-weight: 500; 
    }
    [data-testid="metric-container"] { 
        background: linear-gradient(135deg, #1a2a4f 0%, #2d5a8c 100%); 
        border: 2px solid #00d4ff; 
        border-radius: 12px; 
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4); 
        padding: 20px; 
    }
    [data-testid="metric-container"] label { 
        color: #00ffff; 
        font-weight: bold; 
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5); 
    }
    [data-testid="metric-container"] [data-testid="metric-value"] { 
        color: #00d4ff; 
        font-size: 2em; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.6); 
    }
    .stSuccess { 
        background-color: rgba(0, 102, 204, 0.2) !important; 
        border: 2px solid #00ccff !important; 
        border-radius: 10px !important; 
        box-shadow: 0 0 15px rgba(0, 204, 255, 0.3) !important; 
    }
    .stInfo { 
        background-color: rgba(0, 153, 204, 0.15) !important; 
        border: 2px solid #0099cc !important; 
        border-radius: 10px !important; 
        box-shadow: 0 0 15px rgba(0, 153, 204, 0.3) !important; 
    }
    .stWarning { 
        background-color: rgba(255, 153, 0, 0.15) !important; 
        border: 2px solid #ff9900 !important; 
        border-radius: 10px !important; 
    }
    .stSelectbox label, .stSlider label, .stRadio label { 
        color: #ffffff !important; 
        font-weight: 600 !important; 
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.4) !important; 
    }
    hr { 
        border: 1px solid rgba(0, 212, 255, 0.3) !important; 
    }
    [data-testid="stDataframe"] { 
        background: linear-gradient(135deg, #1a2a4f 0%, #2d5a8c 100%) !important; 
        border: 1px solid #00d4ff !important; 
        border-radius: 10px !important; 
    }
    .stCaption { 
        color: #00ccff !important; 
        text-shadow: 0 0 5px rgba(0, 204, 255, 0.4) !important; 
    }
    [data-testid="stSidebar"] h2 { 
        color: #00ffff !important; 
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6) !important; 
        border-bottom: 2px solid #00d4ff !important; 
        padding-bottom: 10px !important; 
    }
    .stRadio > label { 
        color: #ffffff !important; 
    }
    .stRadio > label > span:first-child { 
        background-color: transparent !important; 
        border: 2px solid #00d4ff !important; 
        border-radius: 6px !important; 
    }
    .stSlider [role="slider"] { 
        background: linear-gradient(90deg, #0099cc, #00ffff) !important; 
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Productivity Gain Index (PGI)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Main Title with Glowing Effect
# ---------------------------
st.markdown("""
    <div class="main-title">
        <div class="main-title-content">
            <h1>📈 Productivity Gain Index – PGI</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 153, 204, 0.08)); border: 2px solid #00d4ff; border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);">
    <p style="color: #ffffff; font-size: 1.05em; line-height: 1.8; margin: 0; font-weight: 500;">
        The <b>Productivity Gain Index (PGI)</b> is a comprehensive analytical framework designed to quantify and visualize the productivity opportunities of nations in an increasingly automated world. By integrating real-world data from the International Labour Organization (ILOSTAT) with advanced mathematical models, our platform reveals how automation speed, wage dynamics, and skill investment collectively influence productivity growth across different countries and sectors.
    </p>
</div>

<p style="text-align: center; color: #00ccff; font-size: 1.08em; margin-top: 15px; margin-bottom: 25px; text-shadow: 0 0 8px rgba(0, 204, 255, 0.5); font-weight: 600;">
Welcome to the Humanoid Work Math Project — explore how automation drives productivity gains and economic growth worldwide.
</p>

st.divider()
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio("Select Mode:", ["Manual Simulation", "ILOSTAT Dataset"])

# Manual Simulation (unchanged)
if mode == "Manual Simulation":
    st.markdown('<div class="section-header"><h2>🧮 PGI Simulation Mode</h2></div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#e0e0e0; font-size:1.03em; margin-bottom:12px;'>Adjust automation speed (A), average earnings (P₀), and elasticity (α) to see expected productivity outcomes.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        A = st.slider("Automation Speed (A)", 0.0, 1.0, 0.4, step=0.05)
    with col2:
        earnings_input = st.number_input("Average Earnings (P₀)", min_value=100.0, max_value=500000.0, value=20000.0, step=100.0, format="%.2f")
    with col3:
        alpha = st.slider("Elasticity (α)", 0.01, 1.0, 0.40, step=0.01)

    st.info(f"Model: P = P₀ × (1 + α × A)  –  α = {alpha:.2f}")

    model = PGIModel(alpha=alpha)
    P = model.compute_pgi_raw(earnings_input, A)
    PGI_pct = model.compute_pgi_percent(earnings_input, A)
    PGI_index_manual = model.compute_pgi_index(earnings_input, A)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("📈 Productivity (P)", f"{P:,.2f}")
    with metric_col2:
        st.metric("📊 Productivity Increase (%)", f"{PGI_pct*100:.2f}%")
    with metric_col3:
        st.metric("📉 PGI (index 0-1)", f"{PGI_index_manual:.3f}")

    st.markdown('<div class="section-header"><h2>📈 PGI Index vs Automation Speed</h2></div>', unsafe_allow_html=True)
    A_values = np.linspace(0, 1, 50)
    P_values = [model.compute_pgi_raw(earnings_input, a) for a in A_values]
    PGI_pct_values = [model.compute_pgi_percent(earnings_input, a) for a in A_values]
    PGI_index_values = [model.compute_pgi_index(earnings_input, a) for a in A_values]

    fig_manual = go.Figure()
    fig_manual.add_trace(go.Scatter(
        x=A_values,
        y=PGI_index_values,
        mode="lines+markers",
        line=dict(color="#00d4ff", width=4),
        marker=dict(size=8, color="#00ccff", line=dict(width=1, color="#ffffff")),
        hovertemplate="A: %{x:.2f}<br>PGI index: %{y:.3f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)"
    ))
    fig_manual.update_layout(
        xaxis=dict(title=dict(text="Automation Speed (A)", font=dict(color="#00d4ff")), tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title=dict(text="PGI Index (0–1)", font=dict(color="#00d4ff")), tickfont=dict(color="#e0e0e0"), range=[0, 1]),
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
    The <b>Productivity Gain Index (PGI)</b> estimates how automation contributes to national productivity.
    We use average earnings as a proxy for baseline productivity (P₀) and model expected productivity as:
    </p>
    """, unsafe_allow_html=True)

    formula_col1, formula_col2 = st.columns(2)
    with formula_col1:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="color:#ff99ff; font-weight:800; margin-bottom:15px;">Base Formula:</h3>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"P = \frac{P_0 \times (1 + \alpha \times A)}{1}")
        st.markdown("""
        <div style="margin-top:20px;">
            <p style="color:#ffffff; font-size:16px; font-weight:600; margin-bottom:12px;">Where:</p>
            <ul style="color:#cbd5e1; font-size:14px; line-height:2.0; list-style-position: inside;">
                <li><b style="color:#ff99ff;">A</b> (Automation Speed) - Rate of technological adoption (0-1)</li>
                <li><b style="color:#ff99ff;">P₀</b> (Baseline Productivity) - Average earnings proxy</li>
                <li><b style="color:#ff99ff;">α</b> (Elasticity) - Productivity sensitivity to automation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with formula_col2:
        st.markdown('<div style="margin-bottom: 20px;"><h3 style="color:#ff66ff; font-weight:800; margin-bottom:15px;">Derived Metrics:</h3></div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:40px;"><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">Percent Gain:</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7; margin-bottom:20px;">ΔP% = (P - P₀) / P₀</p></div>', unsafe_allow_html=True)
        st.markdown('<div><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">PGI Index (0-1):</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7; margin-bottom:20px;">Normalized productivity gain relative to elasticity</p></div>', unsafe_allow_html=True)
        st.markdown('<div><p style="color:#e6f7ff; font-size:15px; font-weight:600; margin-bottom:12px;">Key Insight:</p><p style="color:#cbd5e1; font-size:14px; line-height:1.7;">Higher automation + investment = greater productivity potential</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:0; height:1px; background:rgba(0,212,255,0.12); margin:30px 0;">
    <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.12), rgba(0, 102, 204, 0.18)); border-left:4px solid #00ffff; padding:20px; border-radius:8px;">
      <h3 style="color:#64b5f6; margin-top:0; margin-bottom:15px;">📘 Interpretation Guide:</h3>
      <ul style="color:#f0f3ff; line-height:2.0; font-size:14px;">
        <li><b style="color:#00ff00;">● PGI 0.0 - 0.33 (🟢 Low Growth):</b> Modest productivity gains; stable but slower economic expansion</li>
        <li><b style="color:#ffaa00;">● PGI 0.33 - 0.67 (🟡 Moderate Growth):</b> Significant productivity increases; strong automation adoption with good outcomes</li>
        <li><b style="color:#00ddff;">● PGI 0.67 - 1.0 (🔵 High Growth):</b> Exceptional productivity gains; optimal automation integration and economic development</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ILOSTAT Dataset mode uses the loader from pgi_data_loader
elif mode == "ILOSTAT Dataset":
    st.markdown('<div class="section-header"><h2>🌍 ILOSTAT PGI Integration</h2></div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #e0e0e0; font-size:1.03em; margin-bottom:12px;'>PGI computed from ILOSTAT (latest year per country). Earnings used as P₀; A computed consistently with the ERI pipeline.</p>", unsafe_allow_html=True)

    data = load_pgi_dataset()

    if data.empty:
        st.error("⚠️ Could not load PGI dataset. Please check your data files.")
    else:
        # latest per country (safely handle Year missing)
        if "Year" in data.columns:
            latest = data.sort_values("Year").drop_duplicates("Area", keep="last")
        else:
            latest = data.copy()

        st.markdown('<div class="section-header"><h2>🌍 Global Productivity Gain Index (Latest Year)</h2></div>', unsafe_allow_html=True)

        # check that country names map to Plotly; warn if some don't
        sample_countries = latest["Area"].dropna().unique().tolist()[:20]
        # try a basic check using plotly offline mapping by attempting choropleth with subset
        try:
            fig_map = px.choropleth(
                latest,
                locations="Area",
                locationmode="country names",
                color="PGI_index",
                color_continuous_scale="Blues",
                range_color=(0, 1),
                hover_name="Area",
                hover_data={
                    "Year": True,
                    "PGI_raw": ":,.0f",
                    "PGI_pct": ":.3f",
                    "A": ":.2f",
                    "Earnings": ":,.0f"
                },
                title="Productivity Gain Index (PGI) – Relative (Latest Year)"
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
                coloraxis_colorbar=dict(title="PGI (index 0–1)")
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning("Map rendering failed (likely due to country name mismatches). Showing table and charts only. Error: " + str(e))
            # show a sample of country names to help user debug
            st.info("Sample country names found: " + ", ".join(sample_countries[:10]))

        st.divider()

        # Top & Bottom countries
        st.markdown('<div class="section-header"><h2>📊 Country Comparison – Highest vs Lowest PGI</h2></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="color:#64b5f6; font-weight:bold; text-align:center;">🔵 Highest PGI Countries</p>', unsafe_allow_html=True)
            top = latest.sort_values(by="PGI_index", ascending=False).head(10)
            fig_top = go.Figure(go.Bar(
                x=top["PGI_index"],
                y=top["Area"],
                orientation="h",
                marker=dict(color=top["PGI_index"], colorscale="Blues", line=dict(width=1, color="#ffffff")),
                text=[f"{(p*100):.2f}%" for p in top["PGI_pct"].fillna(0)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>PGI pct: %{text}<br>Index: %{x:.3f}<extra></extra>"
            ))
            fig_top.update_layout(
                height=420,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="PGI Index", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15,20,25,0.8)",
                paper_bgcolor="rgba(26,31,46,0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            st.markdown('<p style="color:#90caf9; font-weight:bold; text-align:center;">🟢 Lowest PGI Countries</p>', unsafe_allow_html=True)
            low = latest.sort_values(by="PGI_index", ascending=True).head(10)
            fig_low = go.Figure(go.Bar(
                x=low["PGI_index"],
                y=low["Area"],
                orientation="h",
                marker=dict(color=low["PGI_index"], colorscale="Blues_r", line=dict(width=1, color="#ffffff")),
                text=[f"{(p*100):.2f}%" for p in low["PGI_pct"].fillna(0)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>PGI pct: %{text}<br>Index: %{x:.3f}<extra></extra>"
            ))
            fig_low.update_layout(
                height=420,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="PGI Index", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15,20,25,0.8)",
                paper_bgcolor="rgba(26,31,46,0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_low, use_container_width=True)

        st.divider()

        # Scatter A vs PGI_index
        st.markdown('<div class="section-header"><h2>⚖️ Automation (A) vs PGI (index)</h2></div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            latest,
            x="A",
            y="PGI_index",
            color="PGI_index",
            color_continuous_scale="Blues",
            range_color=(0, 1),
            hover_name="Area",
            hover_data={"PGI_pct": ":.3f", "Earnings": ":,.0f"},
            labels={"A": "Automation Speed (A)", "PGI_index": "PGI (index 0–1)"},
            title="Automation Speed vs Productivity Gain (relative index)"
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
        st.markdown('<div class="section-header"><h2>🧾 Full PGI Data Table</h2></div>', unsafe_allow_html=True)
        st.dataframe(
            latest[["Area", "Year", "A", "Earnings", "PGI_raw", "PGI_pct", "PGI_index"]].sort_values(by="PGI_index", ascending=False),
            use_container_width=True,
            height=450
        )

# Footer
st.divider()
st.markdown("""
<p style="text-align: center; color: #00ccff; font-size: 0.9em; margin-top: 20px; text-shadow: 0 0 8px rgba(0, 204, 255, 0.4);">
Built with ❤️ by the <b>Humanoid Robots and Future of Work Team (CUT)</b> – Productivity Gain Index (PGI) analysis.
</p>
""", unsafe_allow_html=True)