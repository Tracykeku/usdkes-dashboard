USD/KES Exchange Rate Forecasting Dashboard
> **A comparative study of ARIMA, Random Forest, and LSTM models for predicting the USD/KES exchange rate - with SHAP interpretability and an interactive Streamlit dashboard.**
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
---
Project Overview
The Kenyan Shilling (KES) experienced one of its most dramatic depreciation cycles in recent history - deteriorating from KES 102 per USD (2018) to an unprecedented peak of KES 160 (January 2024), before recovering to approximately KES 130 by April 2026. This volatility has wide-reaching consequences for monetary policy, import costs, debt servicing, and financial risk management.
This project addresses two compounding problems in existing literature:
Predictive gap - no prior study simultaneously benchmarked ARIMA, Random Forest, and LSTM on the USD/KES pair under identical experimental conditions
Interpretability gap - no study had applied SHAP (SHapley Additive exPlanations) to explain what drives USD/KES forecasting model predictions
---
Key Results
Model	MAE (KES)	RMSE (KES)	MAPE (%)	Dir. Accuracy (%)
ARIMA(0,1,1) [BEST]	0.3804	0.5257	0.2941	26.32
LSTM	3.1820	3.6923	2.4611	26.32
Random Forest	7.6285	7.7450	5.9046	31.58
> **ARIMA(0,1,1)** achieved the best accuracy - with forecasts accurate to within **0.29%** of the actual exchange rate on average. It outperformed LSTM by **88%** and Random Forest by **95%** on Mean Absolute Error.
---
SHAP Interpretability - Primary Novel Contribution
This is the first application of SHAP to USD/KES exchange rate forecasting in the literature.
Feature	RF SHAP Weight	LSTM SHAP Weight
USD/KES Lag History	94.3%	45.1%
Brent Crude Oil	2.9%	27.8%
US Dollar Index (DXY)	2.8%	27.1%
Finding: The Random Forest almost entirely ignored macroeconomic signals (only 5.7% combined for Oil + DXY), while the LSTM genuinely integrated them (54.9% combined). This reveals that architectural choice governs not just accuracy but economic awareness.
---
Dashboard Features
The interactive Streamlit dashboard has 7 pages:
Page	Description
Overview	Historical time series, KPI cards, correlation matrix
Model Comparison	Side-by-side metrics, combined forecast chart
Individual Forecasts	Per-model drill-down, residuals, CSV download
Future Forecast	Live forecasting for upcoming months using actual trained models
SHAP & Feature Importance	RF vs LSTM interpretability comparison
Error Analysis	Residuals, scatter plots, error distribution
Notebook Plots	Gallery of all training and evaluation figures
---
Repository Structure
```
usd-kes-dashboard/
dashboard.py              # Main Streamlit application (7 pages)
requirements.txt          # Python dependencies
data/
USD_KES_Master_Dataset.csv    # 100 monthly obs (Jan 2018-Apr 2026)
all_predictions.csv           # Actual vs. predicted (all 3 models)
model_results_full.csv        # MAE, RMSE, MAPE, DA per model
rf_feature_importance.csv     # Random Forest MDI importances
shap_lstm.csv                 # LSTM SHAP values
best_rf_model.pkl             # Trained RF model (joblib)
lstm_model.keras              # Trained LSTM network weights
plots/
01_raw_time_series.png
02_correlation_matrix.png
03_stationarity_check.png
04_arima_forecast.png
05_rf_forecast.png
06_rf_feature_importance.png
07_lstm_training_history.png
08_lstm_forecast.png
09_model_comparison.png
11_shap_rf_bar.png
12_shap_rf_beeswarm.png
13_shap_lstm_bar.png
```
---
Installation & Running Locally
1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/usd-kes-dashboard.git
cd usd-kes-dashboard
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the dashboard
```bash
streamlit run dashboard.py
```
The dashboard will open automatically at `http://localhost:8501`
> **Note:** To enable live Random Forest and LSTM forecasting on the Future Forecast page, ensure `best_rf_model.pkl` and `lstm_model.keras` are present in the `data/` folder.
---
Tech Stack
Category	Tools
Language	Python 3.11
Data	pandas, NumPy, yfinance
Statistics	statsmodels (ADF test, ARIMA)
ML	scikit-learn (Random Forest, MinMaxScaler, GridSearchCV)
Deep Learning	TensorFlow / Keras (LSTM, EarlyStopping)
Interpretability	SHAP (TreeExplainer + GradientExplainer)
Dashboard	Streamlit, Plotly
Model Persistence	joblib (.pkl), Keras (.keras)
Environment	Google Colab Pro (GPU for LSTM training), Anaconda (local)
---
Methodology Summary
Dataset: 100 monthly observations, January 2018 - April 2026
Variables: USD/KES closing rate (CBK) + Brent Crude Oil + US Dollar Index (DXY) from Yahoo Finance
Train/Test Split: 80/20 chronological (Jan 2018-Aug 2024 / Sep 2024-Apr 2026)
Validation: Walk-forward (rolling-origin) - generates 1-step-ahead forecast, reveals true value, updates history, repeats
ARIMA: auto_arima → ARIMA(0,1,1), AIC = 376.531, re-fitted at each test step
Random Forest: 8-feature lag matrix (KES at lags 1,3,6,12 + Brent×2 + DXY×2), GridSearchCV + TimeSeriesSplit(5)
LSTM: Stacked 2-layer (64→32 units), Dropout(0.2), 12-month lookback window, EarlyStopping (best epoch = 5)
SHAP: TreeExplainer for RF (exact Shapley values), GradientExplainer for LSTM (approximation)
---
Data Sources
Variable	Source
USD/KES Exchange Rate	Central Bank of Kenya (CBK)
Brent Crude Oil Price	Yahoo Finance via `yfinance` (ticker: `BZ=F`)
US Dollar Index (DXY)	Yahoo Finance via `yfinance` (ticker: `DX-Y.NYB`)
---
Author
Tracy Nasieku Katimoh
BSc. Data Science and Analytics - Jomo Kenyatta University of Agriculture and Technology (JKUAT), 2026
Registration No: SCT213-C002-0001/2022
---
Acknowledgements
Supervisor: Mr. Samuel Adhola, School of Computing and Information Technology, JKUAT
Central Bank of Kenya for publicly accessible exchange rate data
The open-source Python community - pandas, scikit-learn, TensorFlow, SHAP, Streamlit
