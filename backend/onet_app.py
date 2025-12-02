# onet_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from onet_data_loader import load_onet_analysis

# ---------------------------
# Custom CSS for Green Theme
# ---------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a2e 0%, #2d5a4f 50%, #1a3a2e 100%);
        border-right: 2px solid #00ff88;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Main Title */
    .main-title {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        padding: 2px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.6), 0 0 60px rgba(0, 204, 102, 0.3);
    }
    
    .main-title-content {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 25px;
        border-radius: 13px;
        text-align: center;
    }
    
    .main-title-content h1 {
        color: #00ff88;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
        font-size: 2.5em;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #009933 0%, #00ff88 100%);
        padding: 2px;
        border-radius: 10px;
        margin-top: 25px;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(0, 153, 51, 0.5), 0 0 40px rgba(0, 255, 136, 0.3);
    }
    
    .section-header h2 {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 15px;
        margin: 0;
        border-radius: 8px;
        color: #00ff99;
        text-shadow: 0 0 10px rgba(0, 255, 153, 0.7);
        font-size: 1.6em;
        font-weight: 800;
    }
    
    .stMarkdownContainer, p, span {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a3a2e 0%, #2d5a4f 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
        padding: 20px;
    }
    
    [data-testid="metric-container"] label {
        color: #00ff99;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(0, 255, 153, 0.5);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #00ff88;
        font-size: 2em;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
    }
    
    /* Info Boxes */
    .stSuccess {
        background-color: rgba(0, 153, 51, 0.2) !important;
        border: 2px solid #00ff88 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3) !important;
    }
    
    .stInfo {
        background-color: rgba(0, 255, 136, 0.15) !important;
        border: 2px solid #00cc66 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Divider */
    hr {
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Dataframe */
    [data-testid="stDataframe"] {
        background: linear-gradient(135deg, #1a3a2e 0%, #2d5a4f 100%) !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar Header */
    [data-testid="stSidebar"] h2 {
        color: #00ff99 !important;
        text-shadow: 0 0 8px rgba(0, 255, 153, 0.6) !important;
        border-bottom: 2px solid #00ff88 !important;
        padding-bottom: 10px !important;
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(90deg, #00cc66, #00ff88) !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="O*NET Occupational Analysis",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Main Title
# ---------------------------
st.markdown("""
    <div class="main-title">
        <div class="main-title-content">
            <h1>💼 O*NET Occupational Analysis</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 204, 102, 0.08)); border: 2px solid #00ff88; border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 255, 136, 0.25);">
    <p style="color: #ffffff; font-size: 1.05em; line-height: 1.8; margin: 0; font-weight: 500;">
        The <b>O*NET Occupational Analysis</b> integrates rich occupation data to assess automation risk, skill transferability, and technology requirements by job. By analyzing detailed task structures, skill requirements, and technology adoption patterns, this module provides granular insights into how different occupations face displacement risks and what skills enable workforce resilience.
    </p>
</div>

<p style="text-align: center; color: #00ff88; font-size: 1.08em; margin-top: 15px; margin-bottom: 25px; text-shadow: 0 0 8px rgba(0, 255, 136, 0.5); font-weight: 600;">
Analyze occupation-level automation risks, skill requirements, and transition opportunities.
</p>

st.divider()
""", unsafe_allow_html=True)

# ---------------------------
# Load O*NET Data
# ---------------------------
st.sidebar.header("🧭 Navigation")
view_mode = st.sidebar.radio("Select View:", [
    "Dashboard Overview",
    "Automation Risk Analysis",
    "Skills & Transferability",
    "Technology Requirements",
    "Detailed Occupations"
])

# Load data
risk_df, skills_df, tech_df = load_onet_analysis()

if risk_df.empty:
    st.error("❌ No O*NET data loaded. Please ensure O*NET CSV files are in the current directory.")
    st.info("📌 Required files: Occupations.csv, Task Statements.csv, Skills.csv, Knowledge.csv, Abilities.csv, Technology Skills.csv")
else:
    # ==============================================================
    # 1️⃣ DASHBOARD OVERVIEW
    # ==============================================================
    if view_mode == "Dashboard Overview":
        st.markdown("""
            <div class="section-header">
                <h2>📊 O*NET Database Overview</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Occupations", len(risk_df))
        with col2:
            high_risk = len(risk_df[risk_df["Automation Risk Score"] > 0.67])
            st.metric("🔴 High Risk Occupations", high_risk)
        with col3:
            med_risk = len(risk_df[(risk_df["Automation Risk Score"] >= 0.33) & (risk_df["Automation Risk Score"] <= 0.67)])
            st.metric("🟡 Medium Risk Occupations", med_risk)
        with col4:
            low_risk = len(risk_df[risk_df["Automation Risk Score"] < 0.33])
            st.metric("🟢 Low Risk Occupations", low_risk)
        
        st.divider()
        
        st.markdown("""
            <div class="section-header">
                <h2>📈 Automation Risk Distribution</h2>
            </div>
        """, unsafe_allow_html=True)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=risk_df["Automation Risk Score"],
            nbinsx=30,
            marker=dict(color="#00ff88", line=dict(width=1, color="#ffffff")),
            hovertemplate="Risk Score: %{x:.3f}<br>Occupations: %{y}<extra></extra>"
        ))
        
        fig_dist.update_layout(
            xaxis=dict(title="Automation Risk Score", tickfont=dict(color="#e0e0e0")),
            yaxis=dict(title="Number of Occupations", tickfont=dict(color="#e0e0e0")),
            plot_bgcolor="rgba(15, 20, 25, 0.8)",
            paper_bgcolor="rgba(26, 31, 46, 0.9)",
            font=dict(color="#e0e0e0"),
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ==============================================================
    # 2️⃣ AUTOMATION RISK ANALYSIS
    # ==============================================================
    elif view_mode == "Automation Risk Analysis":
        st.markdown("""
            <div class="section-header">
                <h2>⚠️ Automation Risk Analysis</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <p style="color: #00ff88; font-weight: bold; font-size: 1.1em; text-align: center;">
            🔴 Highest Automation Risk
            </p>
            """, unsafe_allow_html=True)
            
            high_risk = risk_df.nlargest(15, "Automation Risk Score")
            fig_high = go.Figure(go.Bar(
                x=high_risk["Automation Risk Score"],
                y=high_risk["Occupation"],
                orientation="h",
                marker=dict(color=high_risk["Automation Risk Score"], colorscale="Reds", 
                           line=dict(width=1, color="#ffffff")),
                text=[f"{v:.3f}" for v in high_risk["Automation Risk Score"]],
                textposition="outside"
            ))
            fig_high.update_layout(
                height=500,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="Risk Score", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_high, use_container_width=True)
        
        with col2:
            st.markdown("""
            <p style="color: #00ff88; font-weight: bold; font-size: 1.1em; text-align: center;">
            🟢 Lowest Automation Risk
            </p>
            """, unsafe_allow_html=True)
            
            low_risk = risk_df.nsmallest(15, "Automation Risk Score")
            fig_low = go.Figure(go.Bar(
                x=low_risk["Automation Risk Score"],
                y=low_risk["Occupation"],
                orientation="h",
                marker=dict(color=low_risk["Automation Risk Score"], colorscale="Greens", 
                           line=dict(width=1, color="#ffffff")),
                text=[f"{v:.3f}" for v in low_risk["Automation Risk Score"]],
                textposition="outside"
            ))
            fig_low.update_layout(
                height=500,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="Risk Score", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_low, use_container_width=True)
    
    # ==============================================================
    # 3️⃣ SKILLS & TRANSFERABILITY
    # ==============================================================
    elif view_mode == "Skills & Transferability":
        st.markdown("""
            <div class="section-header">
                <h2>🎓 Skills & Transferability Analysis</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if not skills_df.empty:
            skill_summary = skills_df["Transferability"].value_counts()
            
            fig_skills = go.Figure(go.Pie(
                labels=skill_summary.index,
                values=skill_summary.values,
                marker=dict(colors=["#00ff88", "#ffaa00", "#ff6b6b"]),
                hovertemplate="<b>%{label}</b><br>Skills: %{value}<extra></extra>"
            ))
            
            fig_skills.update_layout(
                height=450,
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_skills, use_container_width=True)
            
            st.divider()
            st.markdown("""
            <div class="section-header">
                <h2>🔍 Top Skills by Transferability</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                skills_df.sort_values("Level Normalized", ascending=False).head(20),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Skills data not available")
    
    # ==============================================================
    # 4️⃣ TECHNOLOGY REQUIREMENTS
    # ==============================================================
    elif view_mode == "Technology Requirements":
        st.markdown("""
            <div class="section-header">
                <h2>🔧 Technology Requirements</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if not tech_df.empty:
            fig_tech = go.Figure(go.Bar(
                x=tech_df["Adoption Normalized"],
                y=tech_df["Technology"],
                orientation="h",
                marker=dict(color=tech_df["Adoption Normalized"], colorscale="Viridis",
                           line=dict(width=1, color="#ffffff")),
                text=[f"{v:.2%}" for v in tech_df["Adoption Normalized"]],
                textposition="outside"
            ))
            
            fig_tech.update_layout(
                height=600,
                yaxis=dict(autorange="reversed", tickfont=dict(color="#e0e0e0")),
                xaxis=dict(title="Adoption Level", tickfont=dict(color="#e0e0e0")),
                plot_bgcolor="rgba(15, 20, 25, 0.8)",
                paper_bgcolor="rgba(26, 31, 46, 0.9)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_tech, use_container_width=True)
        else:
            st.warning("⚠️ Technology data not available")
    
    # ==============================================================
    # 5️⃣ DETAILED OCCUPATIONS
    # ==============================================================
    elif view_mode == "Detailed Occupations":
        st.markdown("""
            <div class="section-header">
                <h2>📋 Detailed Occupation Data</h2>
            </div>
        """, unsafe_allow_html=True)
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            min_risk = st.slider(
                "Min Risk Score:",
                0.0, 1.0, 0.0, 0.05
            )
        
        with filter_col2:
            max_risk = st.slider(
                "Max Risk Score:",
                0.0, 1.0, 1.0, 0.05
            )
        
        with filter_col3:
            sort_by = st.selectbox(
                "Sort by:",
                ["Automation Risk Score", "Routine Intensity", "Cognitive Complexity"]
            )
        
        # Apply filters
        filtered_df = risk_df[
            (risk_df["Automation Risk Score"] >= min_risk) & 
            (risk_df["Automation Risk Score"] <= max_risk)
        ].copy()
        
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
        
        st.dataframe(
            filtered_df[["Occupation", "O*NET Code", "Automation Risk Score", 
                        "Routine Intensity", "Manual Intensity", "Cognitive Complexity", 
                        "Human Interaction", "Risk Level"]],
            use_container_width=True,
            height=600
        )

# ==============================================================
# Footer
# ==============================================================
st.divider()
st.markdown("""
<p style="text-align: center; color: #00ff88; font-size: 0.9em; margin-top: 20px; text-shadow: 0 0 8px rgba(0, 255, 136, 0.4);">
Built with ❤️ by the <b>Humanoid Robots and Future of Work Team (CUT)</b> — O*NET Integration for Occupational Analysis.
</p>
""", unsafe_allow_html=True)