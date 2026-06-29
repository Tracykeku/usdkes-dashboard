#
# USD/KES Exchange Rate Forecasting Dashboard
# Tracy Nasieku Katimoh | JKUAT BSc Data Science 2026
# Run: streamlit run dashboard.py
#

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="USD/KES Forecasting Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Global CSS — PRESENTATION MODE (large fonts, high contrast) 
st.markdown("""
<style>
/* ── Base font — large and readable from afar ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px !important;
}

/* ── All text elements — BOLD ── */
p, li, span, div, label {
    font-size: 17px !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
}

/* ── Streamlit markdown text ── */
.stMarkdown p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #0a0a0a !important;
    line-height: 1.7;
}

/* ── Page titles — PROTECTED, cannot be overridden ── */
h1, .stTitle, [data-testid="stTitle"],
div[data-testid="stMarkdownContainer"] h1 {
    font-size: 42px !important;
    font-weight: 900 !important;
    color: #1F3864 !important;
    margin-bottom: 10px !important;
    line-height: 1.2 !important;
}

/* ── Subtitles ── */
h2, h3,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #1F3864 !important;
    line-height: 1.3 !important;
}

/* ── Section title ── */
.section-title {
    font-size: 24px !important;
    font-weight: 900 !important;
}

/* ── Streamlit page title element ── */
.css-10trblm, .css-1629p8f, [class*="css"] h1 {
    font-size: 42px !important;
    font-weight: 900 !important;
    color: #1F3864 !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    color: white;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.kpi-label {
    font-size: 14px !important;
    color: #c8d8e8 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 700 !important;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 40px !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 14px !important;
    color: #a8c8e8 !important;
    margin-top: 6px;
    font-weight: 600 !important;
}

/* ── Section title ── */
.section-title {
    font-size: 24px !important;
    font-weight: 900 !important;
    color: #1F3864 !important;
    border-left: 6px solid #e63946;
    padding-left: 14px;
    margin: 24px 0 16px 0;
    letter-spacing: 0.3px;
}

/* ── Finding / insight box ── */
.finding-box {
    background: #e8f4fd;
    border-left: 6px solid #2196F3;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 17px !important;
    font-weight: 800 !important;
    color: #0a2744 !important;
    line-height: 1.6;
}

/* ── Winner badge ── */
.winner-badge {
    display: inline-block;
    background: #1E6B3C;
    color: white;
    font-size: 13px !important;
    font-weight: 800 !important;
    border-radius: 12px;
    padding: 3px 12px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Dataframe text ── */
.stDataFrame {
    font-size: 16px !important;
}
.stDataFrame th {
    font-size: 16px !important;
    font-weight: 800 !important;
    background-color: #1F3864 !important;
    color: white !important;
}
.stDataFrame td {
    font-size: 16px !important;
    font-weight: 800 !important;
}

/* ── Selectbox and widgets ── */
.stSelectbox label, .stRadio label {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #1F3864 !important;
}

/* ── Download button ── */
.stDownloadButton button {
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
}

/* ── Caption text ── */
.stCaption {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1F3864 !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 18px !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #4a6fa5;
}

/* ── Sidebar navigation active item ── */
[data-testid="stSidebar"] [aria-checked="true"] + div {
    color: #FFD700 !important;
    font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)

#  Colour palette 
COLORS = {
    "ARIMA(0, 1, 1)": "#e63946",
    "Random Forest":  "#f4a261",
    "LSTM":           "#2a9d8f",
    "Actual":         "#1F3864",
    "Training":       "#2196F3",
}

#  Chart font settings — large for projection 
CHART_FONT = dict(family="Segoe UI Bold", size=16, color="#0a0a0a")
AXIS_TITLE_FONT = dict(size=17, color="#000000", family="Segoe UI Bold")
TICK_FONT = dict(size=15, color="#000000", family="Segoe UI Bold")
LEGEND_FONT = dict(size=16, color="#000000", family="Segoe UI Bold")
ANNOTATION_FONT = dict(size=15, color="#333333", family="Segoe UI")

#  Data loader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    d = os.path.join(BASE_DIR, "data")
    df      = pd.read_csv(f"{d}/USD_KES_Master_Dataset.csv",  parse_dates=["Date"], index_col="Date")
    preds   = pd.read_csv(f"{d}/all_predictions.csv",         parse_dates=["Date"], index_col="Date")
    metrics = pd.read_csv(f"{d}/model_results_full.csv",      index_col="Model")
    rf_imp  = pd.read_csv(f"{d}/rf_feature_importance.csv")
    shap_df = pd.read_csv(f"{d}/shap_lstm.csv")
    return df, preds, metrics, rf_imp, shap_df

@st.cache_data
def load_image(filename):
    p = os.path.join(BASE_DIR, "plots", filename)
    with open(p, "rb") as f:
        return f.read()

try:
    df, preds, metrics, rf_imp, shap_lstm = load_data()
except FileNotFoundError as e:
    st.error(f" Could not find data files. Make sure the **data/** folder is next to dashboard.py.\n\nMissing: {e}")
    st.stop()

# ── Helper: apply clean layout to any plotly figure ─────────
def clean_layout(fig, height=460, title=None, legend_below=False):
    updates = dict(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=CHART_FONT,
        margin=dict(l=60, r=30, t=60 if title else 40, b=80),
        xaxis=dict(
            showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc",
            tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc",
            tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT,
        ),
        hovermode="x unified",
        hoverlabel=dict(font_size=15, font_family="Segoe UI"),
        legend=dict(font=LEGEND_FONT),
    )
    if title:
        updates["title"] = dict(text=f"<b>{title}</b>", font=dict(size=20, color="#1F3864", family="Segoe UI"))
    if legend_below:
        updates["legend"] = dict(orientation="h", y=-0.28, x=0.5, xanchor="center", font=LEGEND_FONT)
    fig.update_layout(**updates)
    return fig

def kpi(col, label, value, sub):
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

def finding(text):
    st.markdown(f"<div class='finding-box'> {text}</div>", unsafe_allow_html=True)

#Pre-compute values 
latest_rate = df["USD_KES"].iloc[-1]
ath         = df["USD_KES"].max()
ath_date    = df["USD_KES"].idxmax().strftime("%b %Y")
atl         = df["USD_KES"].min()
atl_date    = df["USD_KES"].idxmin().strftime("%b %Y")
total_depr  = ((latest_rate - df["USD_KES"].iloc[0]) / df["USD_KES"].iloc[0]) * 100
train_df    = df[df.index < preds.index[0]]
test_tail   = train_df.iloc[-24:]

# 
# SIDEBAR
# 
with st.sidebar:
    st.markdown("##  USD/KES Dashboard")
    st.markdown("**ARIMA · Random Forest · LSTM**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "  Overview",
            "  Model Comparison",
            "  Individual Forecasts",
            "  Future Forecast",
            " SHAP & Feature Importance",
            " Error Analysis",
            " Notebook Plots",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Jan 2018 – Apr 2026")
    st.markdown("100 monthly observations")
    st.markdown("**Split:** 80 train / 20 test")
    st.markdown("**Test period:**")
    st.markdown("Sep 2024 – Apr 2026")
    st.markdown("---")
    st.markdown("Tracy Nasieku Katimoh")
    st.markdown("BSc Data Science · JKUAT · 2026")


#
# PAGE 1 — OVERVIEW
# 
if page == "  Overview":

    st.title(" USD/KES Exchange Rate — Historical Overview")
    st.markdown(
        "Comparative study of **ARIMA**, **Random Forest**, and **LSTM** for forecasting the "
        "USD/KES exchange rate using **100 monthly observations** (January 2018 – April 2026)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Latest USD/KES",     f"{latest_rate:.2f}",  "April 2026")
    kpi(c2, "All-Time High",      f"{ath:.2f}",          ath_date)
    kpi(c3, "All-Time Low",       f"{atl:.2f}",          atl_date)
    kpi(c4, "Total Depreciation", f"+{total_depr:.1f}%", "Jan 2018 → Apr 2026")
    kpi(c5, "Observations",       "100",                 "Monthly (CBK + Yahoo)")

    st.markdown("<br>", unsafe_allow_html=True)

    section("Full Time Series — USD/KES, Brent Crude & DXY")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "<b>USD/KES Exchange Rate (KES per USD)</b>",
            "<b>Brent Crude Oil Price (USD/barrel)</b>",
            "<b>US Dollar Index (DXY)</b>",
        ],
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["USD_KES"],     name="USD/KES",     line=dict(color=COLORS["Training"],       width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Brent_Crude"], name="Brent Crude", line=dict(color=COLORS["Random Forest"],  width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DXY"],         name="DXY",         line=dict(color=COLORS["LSTM"],           width=3)), row=3, col=1)

    fig.add_shape(type="rect", x0="2020-02-28", x1="2020-06-30", y0=0, y1=1,
                  xref="x", yref="paper", fillcolor="rgba(255,200,0,0.10)", layer="below", line_width=0)
    fig.add_shape(type="line", x0="2024-01-31", x1="2024-01-31", y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x="2024-01-31", y=0.97, xref="x", yref="paper",
                       text="<b>ATH Jan 2024</b>", showarrow=False,
                       font=dict(color="red", size=15, family="Segoe UI Bold"),
                       xanchor="left", xshift=6)

    fig.update_layout(
        height=640, plot_bgcolor="white", paper_bgcolor="white",
        font=CHART_FONT,
        legend=dict(orientation="h", y=1.04, font=LEGEND_FONT),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    for i in [1, 2, 3]:
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", tickfont=TICK_FONT, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT, row=i, col=1)
    fig.update_annotations(font=dict(size=16, family="Segoe UI", color="#1F3864"))
    st.plotly_chart(fig, use_container_width=True)

    finding("The USD/KES rate depreciated ~57% from KES 99.85 (Feb 2019) to a peak of KES 160.00 (Jan 2024), "
            "then recovered sharply to ~KES 128–130. Brent Crude and DXY show clear co-movement with KES throughout.")

    ca, cb = st.columns(2)
    with ca:
        section("Pearson Correlation Matrix")
        corr = df[["USD_KES","Brent_Crude","DXY"]].corr().round(3)
        fig_c = px.imshow(
            corr, text_auto=True,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_c.update_layout(
            height=360, margin=dict(t=30, b=20), paper_bgcolor="white",
            font=CHART_FONT,
        )
        fig_c.update_traces(textfont=dict(size=18, color="white", family="Segoe UI"))
        st.plotly_chart(fig_c, use_container_width=True)
        finding("DXY has the strongest correlation with USD/KES (r = 0.645). Brent Crude: r = 0.387.")

    with cb:
        section("Descriptive Statistics")
        desc = df[["USD_KES","Brent_Crude","DXY"]].describe().round(3)
        desc.index = ["Count","Mean","Std Dev","Min","25%","Median","75%","Max"]
        st.dataframe(desc, use_container_width=True, height=320)
        finding("USD/KES mean = 118.20 KES, range = 60.15 KES over 100 months — "
                "reflecting the full depreciation-recovery cycle.")


# 
# PAGE 2 — MODEL COMPARISON
# 
elif page == "  Model Comparison":

    st.title("Model Performance Comparison")
    st.markdown("**Test set: Sep 2024 – Apr 2026 · 20 monthly out-of-sample observations · Walk-forward validation**")

    section("Performance Metrics — All Models")

    def highlight_best(s):
        if s.name in ["MAE", "RMSE", "MAPE (%)"]:
            best = s == s.min()
        else:
            best = s == s.max()
        return ["background-color:#1E6B3C; font-weight:900; color:white; font-size:16px" if v
                else "font-size:16px; font-weight:600" for v in best]

    styled = metrics.style.apply(highlight_best).format("{:.4f}")
    st.dataframe(styled, use_container_width=True, height=160)
    st.markdown("** Green = best value per metric**")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    model_order = [("ARIMA(0, 1, 1)", c1, "#e63946", "1"),
                   ("LSTM",           c2, "#2a9d8f", "2"),
                   ("Random Forest",  c3, "#f4a261", "3")]

    for model, col, color, medal in model_order:
        mae  = metrics.loc[model, "MAE"]
        rmse = metrics.loc[model, "RMSE"]
        mape = metrics.loc[model, "MAPE (%)"]
        da   = metrics.loc[model, "Dir. Accuracy (%)"]
        winner = " <span class='winner-badge'>BEST MODEL</span>" if medal == "winner " else ""
        col.markdown(
            f"<div style='border-left:6px solid {color}; background:#f0f4ff; "
            f"border-radius:10px; padding:20px 18px; margin-bottom:10px; "
            f"box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>"
            f"<div style='font-size:22px; font-weight:900; color:{color};'>{medal} {model}{winner}</div>"
            f"<div style='font-size:17px; margin-top:14px; color:#111; line-height:2;'>"
            f"MAE: <b style='font-size:19px'>{mae:.4f} KES</b><br>"
            f"RMSE: <b style='font-size:19px'>{rmse:.4f} KES</b><br>"
            f"MAPE: <b style='font-size:19px'>{mape:.2f}%</b><br>"
            f"Dir. Accuracy: <b style='font-size:19px'>{da:.2f}%</b>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section("Forecast vs Actual — All Three Models")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_tail.index, y=test_tail["USD_KES"],
        name="Training (last 24m)", line=dict(color=COLORS["Training"], width=2.5)
    ))
    fig2.add_trace(go.Scatter(
        x=preds.index, y=preds["Actual"],
        name="Actual", line=dict(color=COLORS["Actual"], width=4)
    ))
    for model, col_key, clr in [
        ("ARIMA(0, 1, 1)", "ARIMA_pred", COLORS["ARIMA(0, 1, 1)"]),
        ("Random Forest",  "RF_pred",    COLORS["Random Forest"]),
        ("LSTM",           "LSTM_pred",  COLORS["LSTM"]),
    ]:
        mae = metrics.loc[model, "MAE"]
        fig2.add_trace(go.Scatter(
            x=preds.index, y=preds[col_key],
            name=f"<b>{model}</b>  (MAE = {mae:.4f})",
            line=dict(color=clr, width=3, dash="dash")
        ))

    split_date = preds.index[0]
    fig2.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                   xref="x", yref="paper", line=dict(color="#555", width=2, dash="dot"))
    fig2.add_annotation(x=split_date, y=0.97, xref="x", yref="paper",
                        text="<b>Test start</b>", showarrow=False,
                        font=dict(color="#000000", size=14, family="Segoe UI Bold"),
                        xanchor="left", xshift=6)

    clean_layout(fig2, height=480, legend_below=True)
    fig2.update_yaxes(title_text="<b>USD/KES (KES per USD)</b>")
    fig2.update_xaxes(title_text="<b>Date</b>")
    st.plotly_chart(fig2, use_container_width=True)

    finding("ARIMA(0,1,1) tracked the actual rate almost exactly. "
            "Random Forest overestimated by ~KES 7 (crisis-era lag anchoring). "
            "LSTM progressively underestimated as it over-extrapolated the recovery trend.")

    c1, c2 = st.columns(2)
    for col, metric, title in [(c1, "MAE", "MAE by Model (lower = better)"),
                                (c2, "RMSE", "RMSE by Model (lower = better)")]:
        with col:
            m_df = metrics.reset_index()
            fig_b = px.bar(
                m_df, x="Model", y=metric,
                color="Model",
                text=m_df[metric].round(4).astype(str),
                color_discrete_map={k: v for k, v in COLORS.items() if k != "Actual"},
            )
            fig_b.update_traces(textposition="outside", textfont=dict(size=16, family="Segoe UI Bold", color="#000000"))
            fig_b.update_layout(
                title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#1F3864")),
                showlegend=False,
                height=400, plot_bgcolor="white", paper_bgcolor="white",
                font=CHART_FONT, margin=dict(t=60, b=40),
            )
            fig_b.update_xaxes(tickfont=dict(size=16, family="Segoe UI Bold", color="#000000"))
            fig_b.update_yaxes(showgrid=True, gridcolor="#e8e8e8",
                               tickfont=dict(size=15, family="Segoe UI Bold", color="#000000"))
            col.plotly_chart(fig_b, use_container_width=True)


# 
# PAGE 3 — INDIVIDUAL FORECASTS
# 
elif page == "  Individual Forecasts":

    st.title(" Individual Model Forecasts")
    st.markdown("**Explore each model's forecast trajectory, monthly errors, and the full prediction table.**")

    model_choice = st.selectbox(
        "Select a model to inspect:",
        ["ARIMA(0, 1, 1)", "Random Forest", "LSTM"],
        index=0
    )

    col_map = {"ARIMA(0, 1, 1)": "ARIMA_pred", "Random Forest": "RF_pred", "LSTM": "LSTM_pred"}
    pcol = col_map[model_choice]
    clr  = COLORS[model_choice]

    mae  = metrics.loc[model_choice, "MAE"]
    rmse = metrics.loc[model_choice, "RMSE"]
    mape = metrics.loc[model_choice, "MAPE (%)"]
    da   = metrics.loc[model_choice, "Dir. Accuracy (%)"]

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "MAE",           f"{mae:.4f}",  "KES avg error")
    kpi(c2, "RMSE",          f"{rmse:.4f}", "KES (penalises outliers)")
    kpi(c3, "MAPE",          f"{mape:.2f}%","% avg error")
    kpi(c4, "Dir. Accuracy", f"{da:.2f}%",  "Correct direction")

    st.markdown("<br>", unsafe_allow_html=True)

    section(f"{model_choice} — Forecast vs Actual")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test_tail.index, y=test_tail["USD_KES"],
        name="Training (last 24m)", line=dict(color=COLORS["Training"], width=2.5)
    ))
    fig3.add_trace(go.Scatter(
        x=preds.index, y=preds["Actual"],
        name="Actual", line=dict(color=COLORS["Actual"], width=4)
    ))
    fig3.add_trace(go.Scatter(
        x=preds.index, y=preds[pcol],
        name=f"{model_choice} Forecast",
        line=dict(color=clr, width=3, dash="dash"),
        fill="tonexty", fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.07)"
    ))
    fig3.add_shape(type="line", x0=preds.index[0], x1=preds.index[0], y0=0, y1=1,
                   xref="x", yref="paper", line=dict(color="#555", width=2, dash="dot"))
    fig3.add_annotation(x=preds.index[0], y=0.97, xref="x", yref="paper",
                        text="<b>Test start</b>", showarrow=False,
                        font=dict(color="#000000", size=14, family="Segoe UI Bold"),
                        xanchor="left", xshift=6)
    clean_layout(fig3, height=460, legend_below=True)
    fig3.update_yaxes(title_text="<b>USD/KES</b>")
    fig3.update_xaxes(title_text="<b>Date</b>")
    st.plotly_chart(fig3, use_container_width=True)

    section("Monthly Residuals  (Actual − Predicted)")
    resid = preds["Actual"] - preds[pcol]
    bar_colors = [clr if r >= 0 else "#e63946" for r in resid]

    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=preds.index, y=resid,
        marker_color=bar_colors,
        text=resid.round(2),
        textposition="outside",
        textfont=dict(size=14, family="Segoe UI Bold", color="#000000"),
    ))
    fig_r.add_hline(y=0, line_dash="dash", line_color="#333", line_width=2)
    clean_layout(fig_r, height=360)
    fig_r.update_yaxes(title_text="<b>Residual (KES)</b>")
    fig_r.update_xaxes(title_text="<b>Date</b>")
    st.plotly_chart(fig_r, use_container_width=True)

    finding(f"Max overestimate: {resid.min():.2f} KES  |  "
            f"Max underestimate: {resid.max():.2f} KES  |  "
            f"Mean residual: {resid.mean():.3f} KES")

    section("Full Prediction Table")
    tbl = pd.DataFrame({
        "Date":        preds.index.strftime("%b %Y"),
        "Actual":      preds["Actual"].round(3),
        "Predicted":   preds[pcol].round(3),
        "Error (A-P)": resid.round(3),
        "Abs Error":   resid.abs().round(3),
        "% Error":     (resid.abs() / preds["Actual"] * 100).round(3),
    }).set_index("Date")

    st.dataframe(tbl, use_container_width=True, height=400)

    csv_data = tbl.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️  Download predictions as CSV",
        data=csv_data,
        file_name=f"{model_choice.replace(' ','_')}_predictions.csv",
        mime="text/csv",
    )


# 
# PAGE — FUTURE FORECAST (live, interactive)
# 
elif page == "  Future Forecast":

    st.title(" Future Forecast — Predict Upcoming Months")
    st.markdown(
        "**Generate live forecasts beyond April 2026.** ARIMA requires no extra input. "
        "Random Forest and LSTM need your expected future Brent Crude Oil and DXY values, "
        "since they are exogenous features the models were trained on."
    )

    #  Try to import optional libraries needed for RF / LSTM 
    rf_available, lstm_available = False, False
    rf_error, lstm_error = "", ""
    try:
        import joblib
        rf_model_path = os.path.join(BASE_DIR, "data", "best_rf_model.pkl")
        if os.path.exists(rf_model_path):
            rf_model = joblib.load(rf_model_path)
            rf_available = True
        else:
            rf_error = "best_rf_model.pkl not found in data/ folder"
    except Exception as e:
        rf_error = str(e)

    try:
        from tensorflow.keras.models import load_model
        lstm_model_path = os.path.join(BASE_DIR, "data", "lstm_model.keras")
        if os.path.exists(lstm_model_path):
            lstm_model = load_model(lstm_model_path, compile=False)
            lstm_available = True
        else:
            lstm_error = "lstm_model.keras not found in data/ folder"
    except Exception as e:
        lstm_error = str(e)

    # Status row 
    c1, c2, c3 = st.columns(3)
    c1.success(" ARIMA — always available (re-fits live on historical data)")
    c2.success("  Random Forest — model loaded") if rf_available else c2.warning(f" Random Forest unavailable: {rf_error}")
    c3.success(" LSTM — model loaded") if lstm_available else c3.warning(f" LSTM unavailable: {lstm_error}")

    if not rf_available or not lstm_available:
        st.info(
            "To enable Random Forest and/or LSTM forecasting, copy **best_rf_model.pkl** and "
            "**lstm_model.keras** from your Google Drive `USD_KES_Project/data_processed/` folder "
            "into this dashboard's **data/** folder, then restart the app. ARIMA works without them."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    #  User controls 
    section("Forecast Settings")
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        n_months = st.slider("How many months ahead?", min_value=1, max_value=12, value=3)

    with ctrl2:
        st.markdown("**Expected future Brent Crude Oil & DXY** *(used by Random Forest and LSTM only)*")
        bc1, bc2 = st.columns(2)
        last_brent = df["Brent_Crude"].iloc[-1]
        last_dxy   = df["DXY"].iloc[-1]
        future_brent = bc1.number_input("Brent Crude Oil (USD/barrel)", value=float(round(last_brent, 2)), step=0.5)
        future_dxy   = bc2.number_input("US Dollar Index (DXY)", value=float(round(last_dxy, 2)), step=0.5)
        st.caption(f"Last known values (Apr 2026): Brent Crude = {last_brent:.2f}, DXY = {last_dxy:.2f}. "
                   "These are held constant across the forecast horizon unless you change them.")

    run = st.button("Generate Forecast", type="primary", use_container_width=True)

    if run:
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_months, freq="ME")
        results = pd.DataFrame(index=future_dates)

        # ARIMA: walk-forward style multi-step forecast 
        with st.spinner("Running ARIMA..."):
            try:
                from statsmodels.tsa.arima.model import ARIMA
                import contextlib, io
                history = list(df["USD_KES"])
                arima_preds = []
                with contextlib.redirect_stdout(io.StringIO()):
                    model = ARIMA(history, order=(0, 1, 1))
                    fitted = model.fit()
                    forecast_res = fitted.forecast(steps=n_months)
                    arima_preds = list(forecast_res)
                results["ARIMA"] = arima_preds
            except Exception as e:
                st.error(f"ARIMA forecasting failed: {e}")
                results["ARIMA"] = np.nan

        #  Random Forest: iterative lag-feature forecasting 
        if rf_available:
            with st.spinner("Running Random Forest..."):
                try:
                    kes_hist = list(df["USD_KES"])
                    rf_preds = []
                    for _ in range(n_months):
                        feat = {
                            "KES_lag1":  kes_hist[-1],
                            "KES_lag3":  kes_hist[-3],
                            "KES_lag6":  kes_hist[-6],
                            "KES_lag12": kes_hist[-12],
                            "Brent_Crude": future_brent,
                            "Brent_lag1":  future_brent,
                            "DXY":         future_dxy,
                            "DXY_lag1":    future_dxy,
                        }
                        X_input = pd.DataFrame([feat])[["KES_lag1","KES_lag3","KES_lag6","KES_lag12",
                                                          "Brent_Crude","Brent_lag1","DXY","DXY_lag1"]]
                        pred = rf_model.predict(X_input)[0]
                        rf_preds.append(pred)
                        kes_hist.append(pred)
                    results["Random Forest"] = rf_preds
                except Exception as e:
                    st.error(f"Random Forest forecasting failed: {e}")
                    results["Random Forest"] = np.nan

        # LSTM: iterative sliding-window forecasting 
        if lstm_available:
            with st.spinner("Running LSTM..."):
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    # Refit scaler on training portion only (Jan 2018 - Aug 2024),
                    # replicating the original notebook's train/test split exactly.
                    train_portion = df.iloc[:80][["USD_KES", "Brent_Crude", "DXY"]]
                    scaler = MinMaxScaler()
                    scaler.fit(train_portion)

                    full_scaled = scaler.transform(df[["USD_KES", "Brent_Crude", "DXY"]])
                    window_data = list(full_scaled[-12:])  # last 12 months, scaled

                    lstm_preds = []
                    for _ in range(n_months):
                        X_input = np.array(window_data[-12:]).reshape(1, 12, 3)
                        pred_scaled = lstm_model.predict(X_input, verbose=0)[0][0]

                        # Inverse-transform just the KES prediction
                        dummy = np.zeros((1, 3))
                        dummy[0, 0] = pred_scaled
                        pred_kes = scaler.inverse_transform(dummy)[0, 0]
                        lstm_preds.append(pred_kes)

                        # Scale the future exogenous inputs the same way for the next window step
                        next_dummy = np.array([[pred_kes, future_brent, future_dxy]])
                        next_scaled = scaler.transform(next_dummy)[0]
                        window_data.append(next_scaled)

                    results["LSTM"] = lstm_preds
                except Exception as e:
                    st.error(f"LSTM forecasting failed: {e}")
                    results["LSTM"] = np.nan

        st.session_state["future_results"] = results
        st.session_state["future_brent"] = future_brent
        st.session_state["future_dxy"] = future_dxy

    #  Display results if available 
    if "future_results" in st.session_state:
        results = st.session_state["future_results"]

        st.markdown("<br>", unsafe_allow_html=True)
        section(f"Forecast Results — Next {len(results)} Month(s)")

        # KPI row of latest known + first/last forecast
        kc1, kc2, kc3 = st.columns(3)
        kpi(kc1, "Last Known Rate", f"{df['USD_KES'].iloc[-1]:.2f}", df.index[-1].strftime("%b %Y"))
        if "ARIMA" in results.columns:
            kpi(kc2, "ARIMA — 1st Month", f"{results['ARIMA'].iloc[0]:.2f}", results.index[0].strftime("%b %Y"))
            kpi(kc3, f"ARIMA — Month {len(results)}", f"{results['ARIMA'].iloc[-1]:.2f}", results.index[-1].strftime("%b %Y"))

        # Chart: history tail + forecasts
        fig_fc = go.Figure()
        hist_tail = df.iloc[-18:]
        fig_fc.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail["USD_KES"],
            name="Historical (last 18m)", line=dict(color=COLORS["Actual"], width=4)
        ))
        connector_x = [df.index[-1]] + list(results.index)
        model_cols = [c for c in results.columns if c in ["ARIMA", "Random Forest", "LSTM"]]
        model_color_map = {"ARIMA": COLORS["ARIMA(0, 1, 1)"], "Random Forest": COLORS["Random Forest"], "LSTM": COLORS["LSTM"]}
        for mcol in model_cols:
            connector_y = [df["USD_KES"].iloc[-1]] + list(results[mcol])
            fig_fc.add_trace(go.Scatter(
                x=connector_x, y=connector_y,
                name=f"<b>{mcol} Forecast</b>",
                line=dict(color=model_color_map[mcol], width=3, dash="dash"),
                mode="lines+markers", marker=dict(size=9)
            ))
        fig_fc.add_shape(type="line", x0=df.index[-1], x1=df.index[-1], y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="#555", width=2, dash="dot"))
        fig_fc.add_annotation(x=df.index[-1], y=0.97, xref="x", yref="paper",
                              text="<b>Forecast starts here</b>", showarrow=False,
                              font=dict(color="#000000", size=14, family="Segoe UI Bold"),
                              xanchor="left", xshift=6)
        clean_layout(fig_fc, height=460, legend_below=True)
        fig_fc.update_yaxes(title_text="<b>USD/KES</b>")
        fig_fc.update_xaxes(title_text="<b>Date</b>")
        st.plotly_chart(fig_fc, use_container_width=True)

        finding(f"Forecast generated using Brent Crude = {st.session_state['future_brent']:.2f} USD/bbl and "
                f"DXY = {st.session_state['future_dxy']:.2f} held constant across the {len(results)}-month horizon "
                f"for Random Forest and LSTM. ARIMA uses only historical USD/KES values.")

        # Table
        section("Forecast Table")
        display_tbl = results.copy()
        display_tbl.index = display_tbl.index.strftime("%b %Y")
        display_tbl = display_tbl.round(3)
        st.dataframe(display_tbl, use_container_width=True)

        st.download_button(
            "  Download forecast as CSV",
            data=results.to_csv().encode("utf-8"),
            file_name="future_forecast.csv",
            mime="text/csv"
        )

        st.warning(
            " **Important caveat:** This forecast assumes Brent Crude Oil and DXY remain at the values "
            "you entered for the entire horizon. Longer horizons (beyond 3 months) carry substantially "
            "more uncertainty, especially for Random Forest and LSTM which rely on historical lag patterns "
            "that may not hold under new macroeconomic conditions."
        )
    else:
        st.info("👆 Set your forecast horizon and expected Brent Crude / DXY values above, then click **Generate Forecast**.")


#
# PAGE 4 — SHAP & FEATURE IMPORTANCE
# 
elif page == " SHAP & Feature Importance":

    st.title(" Feature Importance & SHAP Analysis")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** quantifies which features actually drove "
        "each model's predictions across the 20-month test set. "
        "This is the **primary novelty** of this study — the first SHAP analysis on USD/KES forecasting."
    )

    c1, c2 = st.columns(2)

    with c1:
        section("Random Forest — MDI Feature Importance")
        rf_sorted = rf_imp.sort_values("Importance")
        rf_sorted["Pct"] = (rf_sorted["Importance"] / rf_sorted["Importance"].sum() * 100).round(1)
        rf_sorted["Group"] = rf_sorted["Feature"].apply(
            lambda x: "KES Own History" if "KES_lag" in x else "Macro (Oil/DXY)"
        )
        fig_r = px.bar(
            rf_sorted, x="Importance", y="Feature", orientation="h",
            color="Group",
            color_discrete_map={"KES Own History": COLORS["Training"], "Macro (Oil/DXY)": COLORS["Random Forest"]},
            text=rf_sorted["Pct"].astype(str) + "%",
        )
        fig_r.update_traces(textposition="outside", textfont=dict(size=15, family="Segoe UI Bold", color="#000000"))
        fig_r.update_layout(
            height=460, showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
            font=CHART_FONT, legend=dict(orientation="h", y=-0.22, font=LEGEND_FONT),
            margin=dict(t=20, b=100),
        )
        fig_r.update_xaxes(showgrid=True, gridcolor="#e8e8e8", tickfont=TICK_FONT)
        fig_r.update_yaxes(tickfont=dict(size=15, family="Segoe UI", color="#111"))
        st.plotly_chart(fig_r, use_container_width=True)
        finding("KES lag features dominate at **~94.3%** of total RF importance. "
                "Brent Crude and DXY together: only **~5.7%** — the RF ignores macro signals.")

    with c2:
        section("LSTM — SHAP Feature Importance")
        total = shap_lstm["SHAP_Value"].sum()
        shap_lstm["Pct"] = (shap_lstm["SHAP_Value"] / total * 100).round(1)
        shap_sorted = shap_lstm.sort_values("SHAP_Value")
        shap_sorted["Group"] = shap_sorted["Feature"].apply(
            lambda x: "KES Own History" if "USD_KES" in x else "Macro (Oil/DXY)"
        )
        fig_l = px.bar(
            shap_sorted, x="SHAP_Value", y="Feature", orientation="h",
            color="Group",
            color_discrete_map={"KES Own History": COLORS["Training"], "Macro (Oil/DXY)": COLORS["LSTM"]},
            text=shap_sorted["Pct"].astype(str) + "%",
        )
        fig_l.update_traces(textposition="outside", textfont=dict(size=15, family="Segoe UI Bold", color="#000000"))
        fig_l.update_layout(
            height=460, showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
            font=CHART_FONT, legend=dict(orientation="h", y=-0.22, font=LEGEND_FONT),
            margin=dict(t=20, b=100),
        )
        fig_l.update_xaxes(showgrid=True, gridcolor="#e8e8e8", tickfont=TICK_FONT,
                            title="<b>Mean |SHAP Value|</b>")
        fig_l.update_yaxes(tickfont=dict(size=15, family="Segoe UI", color="#111"))
        st.plotly_chart(fig_l, use_container_width=True)
        finding("LSTM distributes importance evenly: **45% KES history**, "
                "**28% Brent Crude**, **27% DXY** — it genuinely learns from macro signals.")

    section("Feature Group Contribution — RF vs LSTM Side by Side")
    cmp = pd.DataFrame({
        "Model":               ["Random Forest", "LSTM"],
        "KES Own History (%)": [94.3,             45.1],
        "Brent Crude (%)":     [2.9,              27.8],
        "DXY (%)":             [2.8,              27.1],
    })
    fig_s = px.bar(
        cmp.melt(id_vars="Model", var_name="Feature Group", value_name="Contribution (%)"),
        x="Model", y="Contribution (%)", color="Feature Group", barmode="stack",
        text_auto=".1f",
        color_discrete_map={
            "KES Own History (%)": COLORS["Training"],
            "Brent Crude (%)":     COLORS["Random Forest"],
            "DXY (%)":             COLORS["LSTM"],
        },
    )
    fig_s.update_traces(textposition="inside", textfont=dict(color="white", size=16, family="Segoe UI Bold"))
    fig_s.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        font=CHART_FONT, legend=dict(orientation="h", y=-0.22, font=LEGEND_FONT),
        yaxis=dict(range=[0, 108], showgrid=True, gridcolor="#e8e8e8",
                   tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT),
        xaxis=dict(tickfont=dict(size=17, family="Segoe UI Bold", color="#000000")),
        margin=dict(b=100, t=30)
    )
    st.plotly_chart(fig_s, use_container_width=True)

    finding("The RF essentially ignores Oil and DXY despite having them as inputs (~6% combined). "
            "The LSTM actively uses them (55% combined) — showing architectural choice drives economic awareness.")

    st.markdown("<br>", unsafe_allow_html=True)
    section("SHAP Plots from Notebook")
    img_c1, img_c2 = st.columns(2)
    with img_c1:
        st.markdown("**RF — SHAP Bar Plot**")
        st.image(load_image("11_shap_rf_bar.png"), use_container_width=True)
    with img_c2:
        st.markdown("**RF — SHAP Beeswarm Plot** *(directionality of feature effects)*")
        st.image(load_image("12_shap_rf_beeswarm.png"), use_container_width=True)
    st.markdown("**LSTM — SHAP Bar Plot**")
    st.image(load_image("13_shap_lstm_bar.png"), use_container_width=True)


# 
# PAGE 5 — ERROR ANALYSIS
#
elif page == "Error Analysis":

    st.title("Error Analysis")
    st.markdown("**Deep dive into how and where each model fails across the 20-month test period.**")

    err_df = pd.DataFrame({
        "Date":   preds.index,
        "Actual": preds["Actual"].values,
        "ARIMA":  (preds["Actual"] - preds["ARIMA_pred"]).values,
        "RF":     (preds["Actual"] - preds["RF_pred"]).values,
        "LSTM":   (preds["Actual"] - preds["LSTM_pred"]).values,
    }).set_index("Date")

    section("Monthly Residuals — All Models Overlaid  (Actual − Predicted)")
    fig_err = go.Figure()
    for model, col, clr in [("ARIMA", "ARIMA", COLORS["ARIMA(0, 1, 1)"]),
                              ("Random Forest", "RF", COLORS["Random Forest"]),
                              ("LSTM", "LSTM", COLORS["LSTM"])]:
        fig_err.add_trace(go.Scatter(
            x=err_df.index, y=err_df[col],
            name=f"<b>{model}</b>", line=dict(color=clr, width=3),
            mode="lines+markers", marker=dict(size=8)
        ))
    fig_err.add_hline(y=0, line_dash="dash", line_color="#333", line_width=2)
    clean_layout(fig_err, height=420, legend_below=True)
    fig_err.update_yaxes(title_text="<b>Residual (KES)</b>")
    st.plotly_chart(fig_err, use_container_width=True)

    finding("ARIMA residuals cluster tightly around zero. "
            "RF residuals are persistently negative (overestimation by KES 6–8). "
            "LSTM starts negative (overestimate) and drifts positive (underestimate) over time.")

    section("Absolute Error Distribution — Box Plot")
    abs_df = err_df[["ARIMA","RF","LSTM"]].abs().melt(var_name="Model", value_name="Abs Error (KES)")
    color_map = {"ARIMA": COLORS["ARIMA(0, 1, 1)"], "RF": COLORS["Random Forest"], "LSTM": COLORS["LSTM"]}
    fig_box = px.box(abs_df, x="Model", y="Abs Error (KES)", color="Model",
                     color_discrete_map=color_map, points="all")
    fig_box.update_traces(marker_size=8)
    fig_box.update_layout(
        height=420, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=CHART_FONT,
        xaxis=dict(tickfont=dict(size=17, family="Segoe UI Bold", color="#000000")),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8", tickfont=TICK_FONT, title_font=AXIS_TITLE_FONT),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    section("Error Summary Statistics")
    summary = pd.DataFrame({
        "Model":            ["ARIMA(0,1,1)", "Random Forest", "LSTM"],
        "Mean Error":       [err_df["ARIMA"].mean(), err_df["RF"].mean(), err_df["LSTM"].mean()],
        "Std of Errors":    [err_df["ARIMA"].std(),  err_df["RF"].std(),  err_df["LSTM"].std()],
        "Max Overestimate": [err_df["ARIMA"].min(),  err_df["RF"].min(),  err_df["LSTM"].min()],
        "Max Underestimate":[err_df["ARIMA"].max(),  err_df["RF"].max(),  err_df["LSTM"].max()],
        "MAE":              [err_df["ARIMA"].abs().mean(), err_df["RF"].abs().mean(), err_df["LSTM"].abs().mean()],
    }).set_index("Model").round(4)

    st.dataframe(summary.style.format("{:.4f}"), use_container_width=True, height=180)

    finding(f"ARIMA mean error = {err_df['ARIMA'].mean():.4f} KES — essentially unbiased. "
            f"RF mean error = {err_df['RF'].mean():.4f} KES — systematic overestimation. "
            f"LSTM mean error = {err_df['LSTM'].mean():.4f} KES — mixed directional bias.")

    section("Actual vs Predicted Scatter  (perfect model = diagonal line)")
    fig_sc = go.Figure()
    lo = min(preds[["Actual","ARIMA_pred","RF_pred","LSTM_pred"]].min())
    hi = max(preds[["Actual","ARIMA_pred","RF_pred","LSTM_pred"]].max())
    fig_sc.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect forecast",
        line=dict(color="#aaa", dash="dot", width=2)
    ))
    for model, col, clr in [("ARIMA", "ARIMA_pred", COLORS["ARIMA(0, 1, 1)"]),
                              ("Random Forest", "RF_pred", COLORS["Random Forest"]),
                              ("LSTM", "LSTM_pred", COLORS["LSTM"])]:
        fig_sc.add_trace(go.Scatter(
            x=preds["Actual"], y=preds[col],
            mode="markers", name=f"<b>{model}</b>",
            marker=dict(color=clr, size=12, opacity=0.9, line=dict(width=1.5, color="white"))
        ))
    clean_layout(fig_sc, height=460, legend_below=True)
    fig_sc.update_xaxes(title_text="<b>Actual USD/KES</b>")
    fig_sc.update_yaxes(title_text="<b>Predicted USD/KES</b>")
    st.plotly_chart(fig_sc, use_container_width=True)

    finding("ARIMA points hug the diagonal. RF sits in a horizontal band above (overestimation). "
            "LSTM drifts below the diagonal in later months (underestimation).")

    full_err = pd.DataFrame({
        "Date":        preds.index.strftime("%b %Y"),
        "Actual":      preds["Actual"].round(3),
        "ARIMA Pred":  preds["ARIMA_pred"].round(3),
        "ARIMA Error": err_df["ARIMA"].round(3),
        "RF Pred":     preds["RF_pred"].round(3),
        "RF Error":    err_df["RF"].round(3),
        "LSTM Pred":   preds["LSTM_pred"].round(3),
        "LSTM Error":  err_df["LSTM"].round(3),
    }).set_index("Date")
    st.download_button(
        "⬇️  Download full error table as CSV",
        data=full_err.to_csv().encode("utf-8"),
        file_name="all_model_errors.csv",
        mime="text/csv"
    )


# 
# PAGE 6 — NOTEBOOK PLOTS
#
elif page == "Notebook Plots":

    st.title("Notebook Plots Gallery")
    st.markdown("**All figures generated during model training and evaluation in the Jupyter notebook.**")

    plots = [
        ("01_raw_time_series.png",      "Figure 1 — Raw Time Series: USD/KES, Brent Crude, DXY (Jan 2018 – Apr 2026)"),
        ("02_correlation_matrix.png",   "Figure 2 — Pearson Correlation Matrix"),
        ("03_stationarity_check.png",   "Figure 3 — Stationarity Check: USD/KES Level vs First Difference"),
        ("04_arima_forecast.png",       "Figure 4 — ARIMA(0,1,1) Walk-Forward Forecast vs Actual"),
        ("05_rf_forecast.png",          "Figure 5 — Random Forest Forecast vs Actual"),
        ("06_rf_feature_importance.png","Figure 6 — Random Forest MDI Feature Importances"),
        ("07_lstm_training_history.png","Figure 7 — LSTM Training History: Train Loss vs Val Loss"),
        ("08_lstm_forecast.png",        "Figure 8 — LSTM Forecast vs Actual"),
        ("09_model_comparison.png",     "Figure 9 — Three-Panel Model Comparison"),
        ("11_shap_rf_bar.png",          "Figure 11 — SHAP Feature Importance: Random Forest (Bar)"),
        ("12_shap_rf_beeswarm.png",     "Figure 12 — SHAP Beeswarm Plot: Random Forest (Directionality)"),
        ("13_shap_lstm_bar.png",        "Figure 13 — SHAP Feature Importance: LSTM (Bar)"),
    ]

    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plots):
                fname, caption_text = plots[i + j]
                with col:
                    st.markdown(f"**{caption_text}**")
                    st.image(load_image(fname), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)

