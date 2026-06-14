# ============================================================
# USD/KES Exchange Rate Forecasting Dashboard
# Tracy Nasieku Katimoh | JKUAT BSc Data Science 2026
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="USD/KES Forecasting Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Overall font ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Metric cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    color: white;
    margin-bottom: 8px;
}
.kpi-label  { font-size: 12px; color: #aab4be; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value  { font-size: 28px; font-weight: 700; color: #ffffff; }
.kpi-sub    { font-size: 11px; color: #778899; margin-top: 4px; }

/* ── Model cards ── */
.model-card {
    border-left: 4px solid var(--accent);
    background: #f8f9fc;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

/* ── Section title ── */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #1a1a2e;
    border-left: 4px solid #e63946;
    padding-left: 10px;
    margin: 20px 0 12px 0;
}

/* ── Winner badge ── */
.winner-badge {
    display: inline-block;
    background: #28a745;
    color: white;
    font-size: 11px;
    font-weight: 600;
    border-radius: 12px;
    padding: 2px 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Finding box ── */
.finding-box {
    background: #eaf4fb;
    border-left: 4px solid #2196F3;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #1a1a2e;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] hr { border-color: #334; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ───────────────────────────────────────────
COLORS = {
    "ARIMA(0, 1, 1)": "#e63946",
    "Random Forest":  "#f4a261",
    "LSTM":           "#2a9d8f",
    "Actual":         "#1a1a2e",
    "Training":       "#2196F3",
}

# ── Data loader (cached) ─────────────────────────────────────
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
    data_ok = True
except FileNotFoundError as e:
    data_ok = False
    st.error(f"❌ Could not find data files. Make sure the **data/** folder is next to dashboard.py.\n\nMissing: {e}")
    st.stop()

# ── Helper: clean plotly layout ──────────────────────────────
def clean_layout(fig, height=420, title=None, legend_below=False):
    updates = dict(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI", size=12, color="#1a1a2e"),
        margin=dict(l=50, r=20, t=50 if title else 30, b=60),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#ddd"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#ddd"),
        hovermode="x unified",
    )
    if title:
        updates["title"] = dict(text=title, font=dict(size=14, color="#1a1a2e"))
    if legend_below:
        updates["legend"] = dict(orientation="h", y=-0.28, x=0.5, xanchor="center")
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
    st.markdown(f"<div class='finding-box'>💡 {text}</div>", unsafe_allow_html=True)

# ── Pre-compute useful values ────────────────────────────────
latest_rate   = df["USD_KES"].iloc[-1]
ath           = df["USD_KES"].max()
ath_date      = df["USD_KES"].idxmax().strftime("%b %Y")
atl           = df["USD_KES"].min()
atl_date      = df["USD_KES"].idxmin().strftime("%b %Y")
total_depr    = ((latest_rate - df["USD_KES"].iloc[0]) / df["USD_KES"].iloc[0]) * 100
train_df      = df[df.index < preds.index[0]]
test_tail     = train_df.iloc[-24:]    # last 24 months of training for charts

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💱 USD/KES Dashboard")
    st.markdown("*ARIMA · Random Forest · LSTM*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📈  Overview",
            "🏆  Model Comparison",
            "📊  Individual Forecasts",
            "🔍  SHAP & Feature Importance",
            "📉  Error Analysis",
            "📸  Notebook Plots",
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
    st.caption("Tracy Nasieku Katimoh")
    st.caption("BSc Data Science · JKUAT · 2026")


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📈  Overview":

    st.title("📈 USD/KES Exchange Rate — Historical Overview")
    st.markdown(
        "Comparative study of **ARIMA**, **Random Forest**, and **LSTM** for forecasting the "
        "USD/KES exchange rate using 100 monthly observations (January 2018 – April 2026)."
    )

    # ── KPI row ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Latest USD/KES",     f"{latest_rate:.2f}",  "April 2026")
    kpi(c2, "All-Time High",      f"{ath:.2f}",          ath_date)
    kpi(c3, "All-Time Low",       f"{atl:.2f}",          atl_date)
    kpi(c4, "Total Depreciation", f"+{total_depr:.1f}%", "Jan 2018 → Apr 2026")
    kpi(c5, "Observations",       "100",                 "Monthly (CBK + Yahoo)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time series ─────────────────────────────────────────
    section("Full Time Series — USD/KES, Brent Crude & DXY")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("USD/KES Exchange Rate (KES per USD)",
                        "Brent Crude Oil Price (USD/barrel)",
                        "US Dollar Index (DXY)"),
        vertical_spacing=0.07,
        row_heights=[0.4, 0.3, 0.3]
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["USD_KES"],     name="USD/KES",     line=dict(color=COLORS["Training"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Brent_Crude"], name="Brent Crude", line=dict(color=COLORS["Random Forest"], width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DXY"],         name="DXY",         line=dict(color=COLORS["LSTM"], width=2)), row=3, col=1)

    # Annotate key events — use shapes directly for full Plotly version compatibility
    fig.add_shape(type="rect",
                  x0="2020-02-28", x1="2020-06-30", y0=0, y1=1,
                  xref="x", yref="paper",
                  fillcolor="rgba(255,200,0,0.08)", layer="below", line_width=0)
    fig.add_shape(type="line",
                  x0="2024-01-31", x1="2024-01-31", y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="red", width=1.5, dash="dash"))
    fig.add_annotation(x="2024-01-31", y=1, xref="x", yref="paper",
                       text="ATH Jan 2024", showarrow=False,
                       font=dict(color="red", size=11),
                       xanchor="left", yanchor="top", xshift=4)

    fig.update_layout(height=600, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Segoe UI", size=12),
                      legend=dict(orientation="h", y=1.02),
                      margin=dict(l=50, r=20, t=60, b=40))
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    finding("The USD/KES rate depreciated ~57% from KES 99.85 (Feb 2019) to a peak of KES 160.00 (Jan 2024), "
            "then recovered sharply to ~KES 128–130, coinciding with CBK monetary policy interventions and "
            "improved FX inflows. Brent Crude and DXY show clear co-movement with KES throughout.")

    # ── Correlation + Stats ──────────────────────────────────
    ca, cb = st.columns(2)
    with ca:
        section("Pearson Correlation Matrix")
        corr = df[["USD_KES","Brent_Crude","DXY"]].corr().round(3)
        fig_c = px.imshow(
            corr, text_auto=True,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            labels=dict(color="r"),
        )
        fig_c.update_layout(height=320, margin=dict(t=20, b=10), paper_bgcolor="white",
                            font=dict(family="Segoe UI"))
        st.plotly_chart(fig_c, use_container_width=True)
        finding("DXY has the strongest correlation with USD/KES (r = 0.645). "
                "Brent Crude shows moderate correlation (r = 0.387).")

    with cb:
        section("Descriptive Statistics")
        desc = df[["USD_KES","Brent_Crude","DXY"]].describe().round(3)
        desc.index = ["Count","Mean","Std Dev","Min","25%","Median","75%","Max"]
        st.dataframe(desc, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        finding("USD/KES mean = 118.20 KES, range = 60.15 KES over 100 months, "
                "reflecting the full depreciation-recovery cycle.")


# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "🏆  Model Comparison":

    st.title("🏆 Model Performance Comparison")
    st.markdown("Test set: **Sep 2024 – Apr 2026** · 20 monthly out-of-sample observations · Walk-forward validation")

    # ── Metrics table ────────────────────────────────────────
    section("Performance Metrics — All Models")

    def highlight_best(s):
        if s.name in ["MAE", "RMSE", "MAPE (%)"]:
            best = s == s.min()
        else:
            best = s == s.max()
        return ["background-color:#d4edda; font-weight:bold; color:#155724" if v else "" for v in best]

    styled = metrics.style.apply(highlight_best).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)
    st.caption("🟢 Green = best value per metric.")

    # ── Summary cards ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    model_order = [("ARIMA(0, 1, 1)", c1, "#e63946", "🥇"),
                   ("LSTM",           c2, "#2a9d8f", "🥈"),
                   ("Random Forest",  c3, "#f4a261", "🥉")]

    for model, col, color, medal in model_order:
        mae  = metrics.loc[model, "MAE"]
        rmse = metrics.loc[model, "RMSE"]
        mape = metrics.loc[model, "MAPE (%)"]
        da   = metrics.loc[model, "Dir. Accuracy (%)"]
        winner = " <span class='winner-badge'>BEST MODEL</span>" if medal == "🥇" else ""
        col.markdown(
            f"<div style='border-left:4px solid {color}; background:#f8f9fc; "
            f"border-radius:8px; padding:14px 16px; margin-bottom:8px;'>"
            f"<div style='font-size:15px; font-weight:700; color:{color};'>{medal} {model}{winner}</div>"
            f"<div style='font-size:13px; margin-top:10px; color:#444;'>"
            f"MAE: <b>{mae:.4f} KES</b><br>"
            f"RMSE: <b>{rmse:.4f} KES</b><br>"
            f"MAPE: <b>{mape:.2f}%</b><br>"
            f"Dir. Accuracy: <b>{da:.2f}%</b>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    # ── Combined forecast chart ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Forecast vs Actual — All Three Models")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_tail.index, y=test_tail["USD_KES"],
        name="Training (last 24m)", line=dict(color=COLORS["Training"], width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=preds.index, y=preds["Actual"],
        name="Actual", line=dict(color=COLORS["Actual"], width=3)
    ))
    for model, col_key, clr in [
        ("ARIMA(0, 1, 1)", "ARIMA_pred", COLORS["ARIMA(0, 1, 1)"]),
        ("Random Forest",  "RF_pred",    COLORS["Random Forest"]),
        ("LSTM",           "LSTM_pred",  COLORS["LSTM"]),
    ]:
        mae = metrics.loc[model, "MAE"]
        fig2.add_trace(go.Scatter(
            x=preds.index, y=preds[col_key],
            name=f"{model}  (MAE = {mae:.4f})",
            line=dict(color=clr, width=2, dash="dash")
        ))

    # Train/test divider — add_shape for Plotly version compatibility
    split_date = preds.index[0]
    fig2.add_shape(type="line",
                   x0=split_date, x1=split_date, y0=0, y1=1,
                   xref="x", yref="paper",
                   line=dict(color="#888", width=1.5, dash="dot"))
    fig2.add_annotation(x=split_date, y=1, xref="x", yref="paper",
                        text="Test start", showarrow=False,
                        font=dict(color="#888", size=11),
                        xanchor="left", yanchor="top", xshift=4)

    clean_layout(fig2, height=450, legend_below=True)
    fig2.update_yaxes(title_text="USD/KES (KES per USD)")
    fig2.update_xaxes(title_text="Date")
    st.plotly_chart(fig2, use_container_width=True)

    finding("ARIMA(0,1,1) tracked the actual rate almost exactly throughout the test period. "
            "Random Forest overestimated by ~KES 7 due to crisis-era lag anchoring. "
            "LSTM progressively underestimated as it over-extrapolated the recovery trend.")

    # ── MAE / RMSE bar charts ────────────────────────────────
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
            fig_b.update_traces(textposition="outside", textfont_size=11)
            fig_b.update_layout(
                title=title, showlegend=False,
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Segoe UI"), margin=dict(t=50, b=40)
            )
            fig_b.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            col.plotly_chart(fig_b, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — INDIVIDUAL FORECASTS
# ════════════════════════════════════════════════════════════
elif page == "📊  Individual Forecasts":

    st.title("📊 Individual Model Forecasts")
    st.markdown("Explore each model's forecast trajectory, monthly errors, and the full prediction table.")

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

    # ── 4 metric cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "MAE",            f"{mae:.4f}",  "KES avg error")
    kpi(c2, "RMSE",           f"{rmse:.4f}", "KES (penalises outliers)")
    kpi(c3, "MAPE",           f"{mape:.2f}%","% avg error")
    kpi(c4, "Dir. Accuracy",  f"{da:.2f}%",  "Correct direction")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart ───────────────────────────────────────
    section(f"{model_choice} — Forecast vs Actual")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test_tail.index, y=test_tail["USD_KES"],
        name="Training (last 24m)", line=dict(color=COLORS["Training"], width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=preds.index, y=preds["Actual"],
        name="Actual", line=dict(color=COLORS["Actual"], width=3)
    ))
    fig3.add_trace(go.Scatter(
        x=preds.index, y=preds[pcol],
        name=f"{model_choice} Forecast",
        line=dict(color=clr, width=2.5, dash="dash"),
        fill="tonexty", fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.07)"
    ))
    fig3.add_shape(type="line",
                   x0=preds.index[0], x1=preds.index[0], y0=0, y1=1,
                   xref="x", yref="paper",
                   line=dict(color="#888", width=1.5, dash="dot"))
    fig3.add_annotation(x=preds.index[0], y=1, xref="x", yref="paper",
                        text="Test start", showarrow=False,
                        font=dict(color="#888", size=11),
                        xanchor="left", yanchor="top", xshift=4)
    clean_layout(fig3, height=420, legend_below=True)
    fig3.update_yaxes(title_text="USD/KES")
    fig3.update_xaxes(title_text="Date")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Residuals bar chart ──────────────────────────────────
    section("Monthly Residuals  (Actual − Predicted)")
    resid = preds["Actual"] - preds[pcol]
    bar_colors = [clr if r >= 0 else "#e63946" for r in resid]

    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=preds.index, y=resid,
        marker_color=bar_colors,
        text=resid.round(2),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_r.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1)
    clean_layout(fig_r, height=320)
    fig_r.update_yaxes(title_text="Residual (KES)")
    fig_r.update_xaxes(title_text="Date")
    st.plotly_chart(fig_r, use_container_width=True)

    finding(f"Max overestimate: {resid.min():.2f} KES  |  "
            f"Max underestimate: {resid.max():.2f} KES  |  "
            f"Mean residual: {resid.mean():.3f} KES")

    # ── Predictions table ────────────────────────────────────
    section("Full Prediction Table")
    tbl = pd.DataFrame({
        "Date":       preds.index.strftime("%b %Y"),
        "Actual":     preds["Actual"].round(3),
        "Predicted":  preds[pcol].round(3),
        "Error (A-P)":resid.round(3),
        "Abs Error":  resid.abs().round(3),
        "% Error":    (resid.abs() / preds["Actual"] * 100).round(3),
    }).set_index("Date")

    st.dataframe(tbl, use_container_width=True)

    # ── Download button ──────────────────────────────────────
    csv_data = tbl.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️  Download predictions as CSV",
        data=csv_data,
        file_name=f"{model_choice.replace(' ','_')}_predictions.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════
# PAGE 4 — SHAP & FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
elif page == "🔍  SHAP & Feature Importance":

    st.title("🔍 Feature Importance & SHAP Analysis")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) quantifies **which features actually drove each model's predictions** "
        "across the 20-month test set. This directly answers Research Question 4."
    )

    # ── Side-by-side importance charts ──────────────────────
    c1, c2 = st.columns(2)

    with c1:
        section("Random Forest — MDI Feature Importance")
        rf_sorted = rf_imp.sort_values("Importance")
        rf_sorted["Pct"] = (rf_sorted["Importance"] / rf_sorted["Importance"].sum() * 100).round(1)
        # Group: lag vs macro
        rf_sorted["Group"] = rf_sorted["Feature"].apply(
            lambda x: "KES Own History" if "KES_lag" in x or "KES_lag" in x else "Macro (Oil/DXY)"
        )
        fig_r = px.bar(
            rf_sorted, x="Importance", y="Feature", orientation="h",
            color="Group",
            color_discrete_map={"KES Own History": COLORS["Training"], "Macro (Oil/DXY)": COLORS["Random Forest"]},
            text=rf_sorted["Pct"].astype(str) + "%",
        )
        fig_r.update_traces(textposition="outside")
        fig_r.update_layout(
            height=420, showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Segoe UI"), legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=80)
        )
        fig_r.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_r, use_container_width=True)
        finding("KES lag features dominate at **~94.3%** of total RF importance. "
                "Brent Crude and DXY together contribute only **~5.7%**.")

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
        fig_l.update_traces(textposition="outside")
        fig_l.update_layout(
            height=420, showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Segoe UI"), legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=80)
        )
        fig_l.update_xaxes(showgrid=True, gridcolor="#f0f0f0", title="Mean |SHAP Value|")
        st.plotly_chart(fig_l, use_container_width=True)
        finding("LSTM distributes importance far more evenly: **45% KES history**, "
                "**28% Brent Crude**, **27% DXY** — proving it genuinely learns from macro signals.")

    # ── Stacked comparison ───────────────────────────────────
    section("Feature Group Contribution — RF vs LSTM Side by Side")
    cmp = pd.DataFrame({
        "Model":            ["Random Forest", "LSTM"],
        "KES Own History (%)": [94.3,          45.1],
        "Brent Crude (%)":  [2.9,              27.8],
        "DXY (%)":          [2.8,              27.1],
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
    fig_s.update_traces(textposition="inside", textfont=dict(color="white", size=12))
    fig_s.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI"), legend=dict(orientation="h", y=-0.25),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(b=80, t=20)
    )
    st.plotly_chart(fig_s, use_container_width=True)

    finding("The RF is essentially an autoregressive model — it ignores Oil and DXY despite having them as inputs. "
            "The LSTM actively integrates macro signals, making it more suited to volatile macro-driven regimes.")

    # ── SHAP plots from notebook ─────────────────────────────
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


# ════════════════════════════════════════════════════════════
# PAGE 5 — ERROR ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📉  Error Analysis":

    st.title("📉 Error Analysis")
    st.markdown("Deep dive into how and where each model fails across the 20-month test period.")

    # ── Build combined residuals dataframe ───────────────────
    err_df = pd.DataFrame({
        "Date":   preds.index,
        "Actual": preds["Actual"].values,
        "ARIMA":  (preds["Actual"] - preds["ARIMA_pred"]).values,
        "RF":     (preds["Actual"] - preds["RF_pred"]).values,
        "LSTM":   (preds["Actual"] - preds["LSTM_pred"]).values,
    }).set_index("Date")

    # ── Residuals over time (all 3 models) ───────────────────
    section("Monthly Residuals — All Models Overlaid  (Actual − Predicted)")
    fig_err = go.Figure()
    for model, col, clr in [("ARIMA", "ARIMA", COLORS["ARIMA(0, 1, 1)"]),
                              ("Random Forest", "RF", COLORS["Random Forest"]),
                              ("LSTM", "LSTM", COLORS["LSTM"])]:
        fig_err.add_trace(go.Scatter(
            x=err_df.index, y=err_df[col],
            name=model, line=dict(color=clr, width=2),
            mode="lines+markers", marker=dict(size=6)
        ))
    fig_err.add_hline(y=0, line_dash="dash", line_color="#333")
    clean_layout(fig_err, height=380, legend_below=True)
    fig_err.update_yaxes(title_text="Residual (KES)")
    st.plotly_chart(fig_err, use_container_width=True)

    finding("ARIMA residuals cluster tightly around zero throughout. "
            "RF residuals are persistently negative (overestimation by KES 6–8). "
            "LSTM starts negative (overestimate) and drifts positive (underestimate) over time.")

    # ── Abs error distribution ───────────────────────────────
    section("Absolute Error Distribution — Box Plot")
    abs_df = err_df[["ARIMA","RF","LSTM"]].abs().melt(var_name="Model", value_name="Abs Error (KES)")
    color_map = {"ARIMA": COLORS["ARIMA(0, 1, 1)"], "RF": COLORS["Random Forest"], "LSTM": COLORS["LSTM"]}
    fig_box = px.box(abs_df, x="Model", y="Abs Error (KES)", color="Model",
                     color_discrete_map=color_map, points="all")
    fig_box.update_layout(
        height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Error summary table ──────────────────────────────────
    section("Error Summary Statistics")
    summary = pd.DataFrame({
        "Model":           ["ARIMA(0,1,1)", "Random Forest", "LSTM"],
        "Mean Error":      [err_df["ARIMA"].mean(), err_df["RF"].mean(), err_df["LSTM"].mean()],
        "Std of Errors":   [err_df["ARIMA"].std(), err_df["RF"].std(), err_df["LSTM"].std()],
        "Max Overestimate":[err_df["ARIMA"].min(), err_df["RF"].min(), err_df["LSTM"].min()],
        "Max Underestimate":[err_df["ARIMA"].max(), err_df["RF"].max(), err_df["LSTM"].max()],
        "MAE":             [err_df["ARIMA"].abs().mean(), err_df["RF"].abs().mean(), err_df["LSTM"].abs().mean()],
    }).set_index("Model").round(4)

    st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

    finding(f"ARIMA mean error = {err_df['ARIMA'].mean():.4f} KES — essentially unbiased. "
            f"RF mean error = {err_df['RF'].mean():.4f} KES — systematic negative bias (overestimation). "
            f"LSTM mean error = {err_df['LSTM'].mean():.4f} KES — mixed directional bias.")

    # ── Actual vs Predicted scatter ──────────────────────────
    section("Actual vs Predicted Scatter  (perfect model = diagonal line)")
    fig_sc = go.Figure()
    lo = min(preds[["Actual","ARIMA_pred","RF_pred","LSTM_pred"]].min())
    hi = max(preds[["Actual","ARIMA_pred","RF_pred","LSTM_pred"]].max())
    fig_sc.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", name="Perfect forecast",
        line=dict(color="#aaa", dash="dot", width=1.5)
    ))
    for model, col, clr in [("ARIMA", "ARIMA_pred", COLORS["ARIMA(0, 1, 1)"]),
                              ("Random Forest", "RF_pred", COLORS["Random Forest"]),
                              ("LSTM", "LSTM_pred", COLORS["LSTM"])]:
        fig_sc.add_trace(go.Scatter(
            x=preds["Actual"], y=preds[col],
            mode="markers", name=model,
            marker=dict(color=clr, size=9, opacity=0.85,
                        line=dict(width=1, color="white"))
        ))
    clean_layout(fig_sc, height=420, legend_below=True)
    fig_sc.update_xaxes(title_text="Actual USD/KES")
    fig_sc.update_yaxes(title_text="Predicted USD/KES")
    st.plotly_chart(fig_sc, use_container_width=True)

    finding("ARIMA points hug the diagonal closely. RF points cluster in a horizontal band above "
            "actual values (persistent overestimation). LSTM points drift below the diagonal in later months.")

    # ── Download ─────────────────────────────────────────────
    full_err = pd.DataFrame({
        "Date":         preds.index.strftime("%b %Y"),
        "Actual":       preds["Actual"].round(3),
        "ARIMA Pred":   preds["ARIMA_pred"].round(3),
        "ARIMA Error":  err_df["ARIMA"].round(3),
        "RF Pred":      preds["RF_pred"].round(3),
        "RF Error":     err_df["RF"].round(3),
        "LSTM Pred":    preds["LSTM_pred"].round(3),
        "LSTM Error":   err_df["LSTM"].round(3),
    }).set_index("Date")
    st.download_button(
        "⬇️  Download full error table as CSV",
        data=full_err.to_csv().encode("utf-8"),
        file_name="all_model_errors.csv",
        mime="text/csv"
    )


# ════════════════════════════════════════════════════════════
# PAGE 6 — NOTEBOOK PLOTS
# ════════════════════════════════════════════════════════════
elif page == "📸  Notebook Plots":

    st.title("📸 Notebook Plots Gallery")
    st.markdown("All figures generated during model training and evaluation in the Jupyter notebook.")

    plots = [
        ("01_raw_time_series.png",   "Figure 1 — Raw Time Series: USD/KES, Brent Crude, DXY (Jan 2018 – Apr 2026)"),
        ("02_correlation_matrix.png","Figure 2 — Pearson Correlation Matrix"),
        ("03_stationarity_check.png","Figure 3 — Stationarity Check: USD/KES Level vs First Difference"),
        ("04_arima_forecast.png",    "Figure 4 — ARIMA(0,1,1) Walk-Forward Forecast vs Actual"),
        ("05_rf_forecast.png",       "Figure 5 — Random Forest Forecast vs Actual"),
        ("06_rf_feature_importance.png","Figure 6 — Random Forest MDI Feature Importances"),
        ("07_lstm_training_history.png","Figure 7 — LSTM Training History: Train Loss vs Val Loss"),
        ("08_lstm_forecast.png",     "Figure 8 — LSTM Forecast vs Actual"),
        ("09_model_comparison.png",  "Figure 9 — Three-Panel Model Comparison"),
        ("11_shap_rf_bar.png",       "Figure 11 — SHAP Feature Importance: Random Forest (Bar)"),
        ("12_shap_rf_beeswarm.png",  "Figure 12 — SHAP Beeswarm Plot: Random Forest (Directionality)"),
        ("13_shap_lstm_bar.png",     "Figure 13 — SHAP Feature Importance: LSTM (Bar)"),
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
