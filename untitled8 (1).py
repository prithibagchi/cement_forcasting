import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title(':red[Forecasting Sales of Cement]')
st.write(":blue[Import the time series CSV file]") 
uploaded_file = st.file_uploader(" ", type=['csv'])

if uploaded_file is not None:     
    cement = pd.read_csv(uploaded_file)
    st.write(uploaded_file)
    
    hwe_model_add_add = ExponentialSmoothing(cement["sales_in_million_rs"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()

    newdata_pred = hwe_model_add_add.predict(start = len(cement), end = len(cement)+11)
    
    st.subheader(":green[Holts winter exponential smoothing with additive seasonality and additive trend]")
   
    st.write("Sales Forecast: ", newdata_pred)
    st.title(':orange[Forecasting Deamand of Cement]')
    hol_we_addmul = ExponentialSmoothing(cement["demand_mmt"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
    forecast_demand = hol_we_addmul.predict(start = len(cement), end = len(cement)+11)
    st.subheader(":indigo[Holts winter exponential smoothing with multiplicative seasonality and additive trend]")
   
    st.write("Forecasted values: ", forecast_demand)
    st.subheader(":blue[Thank you]")




