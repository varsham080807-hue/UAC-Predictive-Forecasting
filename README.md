Predictive Forecasting of Care Load & Placement Demand
Project Overview
This project forecasts the number of children in HHS care and estimates short-term discharge demand using daily UAC Program data.
Features
Time-series data preparation
Missing-value handling and daily continuity
Lag and rolling features
Naive and moving-average baselines
Exponential Smoothing
Random Forest and Gradient Boosting models
Forecast visualization and approximate uncertainty intervals
Capacity-stress early warning indicator
Streamlit interactive dashboard
Dataset Columns
Date
Children apprehended and placed in CBP custody
Children in CBP custody
Children transferred out of CBP custody
Children in HHS Care
Children discharged from HHS Care
Run Locally
pip install -r requirements.txt
streamlit run app.py
Deployment
Push this repository to GitHub and deploy it using Streamlit Community Cloud.
Important Note
This dashboard is a decision-support and academic forecasting prototype. Forecast uncertainty should be reviewed before operational use.
