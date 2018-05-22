#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 02:32:35 2018

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 06:09:27 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

from fbprophet import Prophet

# Read data
df = pd.read_csv("UberKM.csv")

df = df[['timestamp','expected_wait_time']]

# drop columns id, timestamp, locationID
#df = df.drop(['id','locationID','surge_multiplier'],axis=1)

df.columns = ['ds', 'y']
# make timestamp an index
df.index = pd.to_datetime(df['ds'])
df = df.dropna()

# resample ETA and multiplier in 3min intervals calculating mean value inside each interval
df_new = df.resample('2min', how='mean') # resampled ETA
df_new['ds'] = df_new.index

df_new = df_new.dropna()

df_train = df_new#df_new[:1200]
df_test = df_new[1200:]
df_test = df_test.drop(['y'],axis=1)

m = Prophet()
m.fit(df_train)

#future = m.make_future_dataframe(periods=14)
#future.tail()

forecast = m.predict(df_train)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast)

#plt.plot(forecast['ds'],forecast['yhat'],color='green')
#plt.plot(forecast['ds'],df_new['y'].iloc[1200:], color='blue')

df_new['forecast'] = forecast[['trend']].values
df_new['y_detrend'] = df_new['y']-df_new['forecast']

plt.plot(df_train['ds'],forecast['yhat'],color='green')
plt.plot(df_train['ds'],df_new['y'], color='blue')

df = pd.read_csv("UberKM.csv")
df = df[['timestamp','surge_multiplier']]

df.columns = ['ds', 'y']
# make timestamp an index
df.index = pd.to_datetime(df['ds'])
df = df.dropna()

# resample ETA and multiplier in 3min intervals calculating mean value inside each interval
df_new_surge = df.resample('2min', how='mean') # resampled ETA
df_new_surge = df_new_surge.dropna()

plt.plot(df_new['y_detrend']/max(df_new['y_detrend']),color='green')
plt.plot(df_new_surge['y'],color='blue')


plt.show()

m.plot_components(forecast)
