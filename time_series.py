# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 18:04:06 2021

@author: yashr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks
import statsmodels.api as sm

df = pd.read_csv("D:\\Transactions-Last365days - job1 (1).csv")

df.category.describe()

df.date.describe()

df['date'] = df['date'].astype('datetime64[ns]')

expenses = df.groupby('date')['amount'].sum().reset_index()
expenses = expenses.set_index('date')

expenses.sort_index(inplace = True)

expenses.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 15, 7
expenses.plot()


from statsmodels.tsa.stattools import adfuller
sales = expenses['amount']


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho),reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(expenses['amount'])


from pmdarima.arima.utils import ndiffs
y = expenses.amount

ndiffs(y, test = 'adf')

ndiffs(y, test = 'kpss')

ndiffs(y, test = 'pp')


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(expenses['amount'])
plt.show()


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(expenses['amount'].dropna(),lags=14,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(expenses['amount'].dropna(),lags=14,ax=ax2)

# from statsmodels.tsa.arima_model import ARIMA
# model=ARIMA(expenses['amount'],order=(1,1,1))
# model_fit=model.fit()
# model_fit.summary()

# expenses['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
# expenses[['amount','forecast']].plot(figsize=(12,8))


# import statsmodels.api as sm
# model=sm.tsa.statespace.SARIMAX(expenses['amount'])
# results=model.fit(disp = 0)
# expenses['forecast']=results.predict(start=90,end=103,dynamic=True)
# expenses[['amount','forecast']].plot(figsize=(12,8))

# from pandas.tseries.offsets import DateOffset
# future_dates=[expenses.index[-1]+ DateOffset(days=x)for x in range(0,24)]
# future_datest_df=pd.DataFrame(index=future_dates[1:],columns=expenses.columns)

# future_datest_df.tail()

# future_df=pd.concat([expenses,future_datest_df])

# future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
# future_df[['amount', 'forecast']].plot(figsize=(12, 8))

from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(expenses.amount, order=(1,0,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model_fit.plot_predict(dynamic=False)
plt.show()
