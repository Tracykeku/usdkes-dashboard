import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="USD/KES Forecasting Dashboard", page_icon="💱", layout="wide")

st.markdown("""
<style>
.metric-card{background:#f0f4f8;border-radius:10px;padding:16px 20px;text-align:center;}
.metric-label{font-size:13px;color:#555;margin-bottom:4px;}
.metric-value{font-size:26px;font-weight:700;color:#1a1a2e;}
.metric-sub{font-size:12px;color:#888;}
.section-title{font-size:20px;font-weight:600;color:#1a1a2e;border-left:4px solid #e63946;padding-left:10px;margin-bottom:16px;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df       = pd.read_csv("data/USD_KES_Master_Dataset.csv", parse_dates=["Date"], index_col="Date")
    preds    = pd.read_csv("data/all_predictions.csv", parse_dates=["Date"], index_col="Date")
    metrics  = pd.read_csv("data/model_results_full.csv", index_col="Model")
    rf_imp   = pd.read_csv("data/rf_feature_importance.csv")
    shap_df  = pd.read_csv("data/shap_lstm.csv")
    return df, preds, metrics, rf_imp, shap_df

df, preds, metrics, rf_imp, shap_lstm = load_data()

COLORS = {"ARIMA(0, 1, 1)":"#e63946","Random Forest":"#f4a261","LSTM":"#2a9d8f","Actual":"#1a1a2e"}

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png", width=80)
    st.markdown("## USD/KES Forecasting")
    st.markdown("*ARIMA vs Random Forest vs LSTM*")
    st.markdown("---")
    page = st.radio("Navigate to", ["📈 Overview","🏆 Model Comparison","🔍 Feature Importance & SHAP","📊 Individual Forecasts"])
    st.markdown("---")
    st.markdown("**Period:** Jan 2018 – Apr 2026")
    st.markdown("**Observations:** 100 monthly")
    st.markdown("**Train / Test:** 80 / 20")
    st.markdown("---")
    st.caption("Tracy Nasieku Katimoh · JKUAT · 2026")

if page == "📈 Overview":
    st.title("📈 USD/KES Exchange Rate — Historical Overview")
    st.markdown("Comparative study of ARIMA, Random Forest and LSTM for forecasting the USD/KES exchange rate using monthly data from January 2018 to April 2026.")
    c1,c2,c3,c4 = st.columns(4)
    for col,label,val,sub in zip([c1,c2,c3,c4],["Latest USD/KES","All-Time High","All-Time Low","Total Depreciation"],["130.05","160.00","99.85","+27.5%"],["April 2026","January 2024","February 2019","Jan 2018 → Apr 2026"]):
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Raw Feature Time Series</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=("USD/KES Exchange Rate","Brent Crude Oil (USD/barrel)","US Dollar Index (DXY)"),vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=df.index,y=df["USD_KES"],name="USD/KES",line=dict(color="#2196F3",width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Brent_Crude"],name="Brent Crude",line=dict(color="#f4a261",width=2)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DXY"],name="DXY",line=dict(color="#2a9d8f",width=2)),row=3,col=1)
    fig.add_vline(x="2024-01-31",line_dash="dash",line_color="red",opacity=0.5,row=1,col=1)
    fig.update_layout(height=620,plot_bgcolor="white",paper_bgcolor="white",legend=dict(orientation="h",y=1.02))
    fig.update_xaxes(showgrid=True,gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True,gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
    ca,cb = st.columns(2)
    with ca:
        st.markdown("<div class='section-title'>Pearson Correlation Matrix</div>", unsafe_allow_html=True)
        fig_c = px.imshow(df.corr().round(3),text_auto=True,color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
        fig_c.update_layout(height=320,margin=dict(t=20))
        st.plotly_chart(fig_c,use_container_width=True)
    with cb:
        st.markdown("<div class='section-title'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(3),use_container_width=True)

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Performance Comparison")
    st.markdown("Test set: **Sep 2024 – Apr 2026** (20 monthly observations)")
    st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
    def highlight_best(s):
        is_best = s==s.min() if s.name in ["MAE","RMSE","MAPE (%)"] else s==s.max()
        return ["background-color:#d4edda;font-weight:bold" if v else "" for v in is_best]
    st.dataframe(metrics.style.apply(highlight_best).format("{:.4f}"),use_container_width=True)
    st.caption("Green = best value per metric.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Forecast vs Actual — All Models</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    tail = df[df.index < preds.index[0]].iloc[-24:]
    fig2.add_trace(go.Scatter(x=tail.index,y=tail["USD_KES"],name="Training (last 24m)",line=dict(color="#2196F3",width=2)))
    fig2.add_trace(go.Scatter(x=preds.index,y=preds["Actual"],name="Actual",line=dict(color="#1a1a2e",width=3)))
    for m,col,clr in [("ARIMA(0, 1, 1)","ARIMA_pred","#e63946"),("Random Forest","RF_pred","#f4a261"),("LSTM","LSTM_pred","#2a9d8f")]:
        fig2.add_trace(go.Scatter(x=preds.index,y=preds[col],name=f"{m} (MAE={metrics.loc[m,'MAE']:.2f})",line=dict(color=clr,width=2,dash="dash")))
    fig2.update_layout(height=420,xaxis_title="Date",yaxis_title="USD/KES",plot_bgcolor="white",paper_bgcolor="white",hovermode="x unified",legend=dict(orientation="h",y=-0.25))
    fig2.update_xaxes(showgrid=True,gridcolor="#f0f0f0")
    fig2.update_yaxes(showgrid=True,gridcolor="#f0f0f0")
    st.plotly_chart(fig2,use_container_width=True)
    c1,c2 = st.columns(2)
    for col,metric,title in zip([c1,c2],["MAE","RMSE"],["MAE by Model (lower=better)","RMSE by Model (lower=better)"]):
        with col:
            fig_b = px.bar(metrics.reset_index(),x="Model",y=metric,color="Model",color_discrete_map={k:v for k,v in COLORS.items() if k!="Actual"},title=title,text_auto=".4f")
            fig_b.update_layout(showlegend=False,height=340,plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig_b,use_container_width=True)

elif page == "🔍 Feature Importance & SHAP":
    st.title("🔍 Feature Importance & SHAP Analysis")
    st.markdown("SHAP quantifies each feature's contribution to model predictions, addressing **Research Question 1**: do Brent Crude and DXY improve forecasts?")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Random Forest — Feature Importance</div>", unsafe_allow_html=True)
        fig_r = px.bar(rf_imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale="Oranges",title="RF Feature Importance (MDI)")
        fig_r.update_layout(height=400,showlegend=False,plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
        st.plotly_chart(fig_r,use_container_width=True)
        st.info("**Finding:** KES lags account for ~94% of RF importance. Oil and DXY contribute only ~6%.")
    with c2:
        st.markdown("<div class='section-title'>LSTM — SHAP Feature Importance</div>", unsafe_allow_html=True)
        total = shap_lstm["SHAP_Value"].sum()
        shap_lstm["Pct"] = (shap_lstm["SHAP_Value"]/total*100).round(1)
        fig_l = px.bar(shap_lstm.sort_values("SHAP_Value"),x="SHAP_Value",y="Feature",orientation="h",color="SHAP_Value",color_continuous_scale="Greens",title="LSTM SHAP Values",text="Pct")
        fig_l.update_traces(texttemplate="%{text}%",textposition="outside")
        fig_l.update_layout(height=400,showlegend=False,plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
        st.plotly_chart(fig_l,use_container_width=True)
        st.info("**Finding:** LSTM uses features more evenly — USD_KES 45%, Brent 28%, DXY 27%.")
    st.markdown("<div class='section-title'>Feature Group Contribution Comparison</div>", unsafe_allow_html=True)
    cmp = pd.DataFrame({"Model":["Random Forest","LSTM"],"Own History (%)": [94.3,45.1],"Brent Crude (%)": [2.9,27.8],"DXY (%)": [2.8,27.1]})
    fig_s = px.bar(cmp.melt(id_vars="Model",var_name="Group",value_name="Contribution (%)"),x="Model",y="Contribution (%)",color="Group",barmode="stack",color_discrete_map={"Own History (%)":"#2196F3","Brent Crude (%)":"#f4a261","DXY (%)":"#2a9d8f"})
    fig_s.update_layout(height=360,plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig_s,use_container_width=True)

elif page == "📊 Individual Forecasts":
    st.title("📊 Individual Model Forecasts")
    model_choice = st.selectbox("Select a model:",["ARIMA(0, 1, 1)","Random Forest","LSTM"])
    col_map = {"ARIMA(0, 1, 1)":"ARIMA_pred","Random Forest":"RF_pred","LSTM":"LSTM_pred"}
    clr = COLORS[model_choice]; pcol = col_map[model_choice]
    mae,rmse,mape,da = metrics.loc[model_choice,["MAE","RMSE","MAPE (%)","Dir. Accuracy (%)"]]
    c1,c2,c3,c4 = st.columns(4)
    for col,label,val,unit in zip([c1,c2,c3,c4],["MAE","RMSE","MAPE","Dir. Accuracy"],[mae,rmse,mape,da],["KES","KES","%","%"]):
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val:.4f}</div><div class='metric-sub'>{unit}</div></div>",unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    tail = df[df.index < preds.index[0]].iloc[-24:]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=tail.index,y=tail["USD_KES"],name="Training (last 24m)",line=dict(color="#2196F3",width=2)))
    fig3.add_trace(go.Scatter(x=preds.index,y=preds["Actual"],name="Actual",line=dict(color="#1a1a2e",width=3)))
    fig3.add_trace(go.Scatter(x=preds.index,y=preds[pcol],name=f"{model_choice} Forecast",line=dict(color=clr,width=2.5,dash="dash")))
    fig3.update_layout(height=400,xaxis_title="Date",yaxis_title="USD/KES",plot_bgcolor="white",paper_bgcolor="white",hovermode="x unified",legend=dict(orientation="h",y=-0.2))
    fig3.update_xaxes(showgrid=True,gridcolor="#f0f0f0")
    fig3.update_yaxes(showgrid=True,gridcolor="#f0f0f0")
    st.plotly_chart(fig3,use_container_width=True)
    resid = preds["Actual"]-preds[pcol]
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(x=preds.index,y=resid,marker_color=[clr if r>=0 else "#e63946" for r in resid]))
    fig_r.add_hline(y=0,line_dash="dash",line_color="black")
    fig_r.update_layout(height=300,xaxis_title="Date",yaxis_title="Residual (KES)",plot_bgcolor="white",paper_bgcolor="white",title="Monthly Residuals")
    fig_r.update_xaxes(showgrid=True,gridcolor="#f0f0f0")
    fig_r.update_yaxes(showgrid=True,gridcolor="#f0f0f0")
    st.plotly_chart(fig_r,use_container_width=True)
    tbl = pd.DataFrame({"Date":preds.index.strftime("%b %Y"),"Actual":preds["Actual"].round(3),"Predicted":preds[pcol].round(3),"Error":resid.round(3),"Abs Error":resid.abs().round(3)})
    st.dataframe(tbl.set_index("Date"),use_container_width=True)
