import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#import seaborn as sns 
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import datetime
from datetime import date,timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#TItle
app_name='Stock Marketing Forecast App'
st.title(app_name)
st.subheader('this app is created to forecast the stock market price of the selected company')
st.image('https://www.livemint.com/lm-img/img/2023/05/29/1600x900/Nifty_1669509049999_1685353797332.jpg')

#Sidebar
st.sidebar.header('Select the parameters from below')
start_date=st.sidebar.date_input('Start date',date(2009,12,31))
end_date=st.sidebar.date_input('End date',date(2017,11,15))

#Add ticker symbol list
ticker_list=['AAPL','DAAA','DBAA','GE','MSFT','WFC','XOM','FTSE','GDAXI']
ticker=st.sidebar.selectbox('select the company',ticker_list)


Nasdaq= pd.read_csv("Processed_NASDAQ.csv")
Nasdaq[['Date']]=Nasdaq[['Date']].apply(pd.to_datetime)
Nasdaq.dropna(axis=1,how='all')
#Nasdaq.insert(0,'ticker',ticker)
#Nasdaq.insert(1,'start_date',start_date)
#Nasdaq.insert(2,'end_date',end_date)
Nasdaq.reset_index(drop=True,inplace=True)
st.write('Data From',start_date,'to',end_date)
st.dataframe(Nasdaq.head(1000))
#st.set_option('deprecation.showPyplotGlobalUse', False)

#plot the data
st.header('Data Visualization')
st.subheader('Plot Of The Data')
fig=px.line(Nasdaq,x='Date',y=['Close'],title='Closing price of the stock',width=1000,height=600)
st.plotly_chart(fig)

#add a select box to select column from data 
column=st.selectbox('select the column to be used for forecasting',Nasdaq.columns)

#subsetting the data 
data=Nasdaq[['Date',column]]
st.write(data)


#ADF test check stationarity
st.header('Is Data Stationary?')
st.write(adfuller(data[column])[1]<0.05)

#Decompose The Data 
st.header('Decompostion of the Data')
decompostion=seasonal_decompose(data[column],model='additive',period=12)
st.write(decompostion.plot())

st.write('## Plotting the decompostion in plotly')
st.plotly_chart(px.line(x=data['Date'],y=decompostion.trend,title='Trend',width=1000,height=400,labels={'x':'Date','y':data.columns[1]}))

#Running The Model 
# user input for three parameters of the model and seasonal order
p=st.slider('Select the value of p',0,5,2)
d=st.slider('Select the value of d',0,5,1)
q=st.slider('Select the value of q',0,5,2)
seasonal_order=st.number_input('Select the value of seasonal p',0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()

#Print the model summary 
st.header('Model Summary')
st.write(model.summary())
st.write("---")

#Predict the future values (Forecasting)
forecast_period=st.number_input('Select the number of days to forecast',1,365,30)
predections=model.get_prediction(start=len(data),end=len(data)+forecast_period)
predections=predections.predicted_mean


#add index to results dataframe as dates
predections.index=pd.date_range(start=end_date,periods=len(predections),freq='D')
predections=pd.DataFrame(predections)
predections.insert(0,'Date',predections.index,True)
predections.reset_index(drop=True,inplace=True)
st.write('## predictions',predections)
st.write('## Actual Data',data)

#Plot The Data 
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predections['Date'],y=predections['predicted_mean'],mode='lines',name='Predicted',line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Price',width=1200,height=400)
st.plotly_chart(fig)



#ADD buttons to show and hide separate plots :
show_plots=False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data['Date'],y=data[column],title='Actual',width=1200,height=400,labels={'x':'Date','y':'Price'}))
        st.write(px.line(x=predections['Date'],y=predections['predicted_mean'],title='predicted',width=1200,height=400,labels={'x':'Date','y':'predict'}))
        show_plots=True
else:
 #(variable) hide_plots: Literal[False]
    hide_plots=False
if st.button('Hide Separate Plots'):
    if not hide_plots:
        hide_plots=True
    else:
        hide_plots=False
    
st.write("-----")







## merge predicted values with actual values based on Date Column 
#combined_df=pd.merge(data,predections,on='Date',how='outer')
#combined_df.columns=['Date','Actual','Predicted']
#combined_df.set_index('Date',inplace=True)
#st.write('## Combined Data',combined_df)







































