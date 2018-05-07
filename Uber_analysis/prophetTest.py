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
df = pd.read_csv("uber_WDC_4weeks.csv")

# drop columns id, timestamp, locationID
df = df.drop(['id','locationID'],axis=1)

df.columns = ['ds', 'y', 'expected_wait_time']

# make timestamp an index
df.index = pd.to_datetime(df['ds'])

# resample ETA and multiplier in 3min intervals calculating mean value inside each interval
df_new = df.resample('3min', how='mean') # resampled ETA

df_new['ds'] = df_new.index

df_new = df_new[['ds','y', 'expected_wait_time']]
df_new = df_new.reset_index(drop=True)
df_new['expected_wait_time'] = np.roll(df_new['expected_wait_time'].values, shift=int(1))

df_new = df_new.dropna()

df_train = df_new[:10000]
df_test = df_new[10000:]
df_test = df_test.drop(['y'],axis=1)

m = Prophet()
m.add_regressor('expected_wait_time')
m.fit(df_train)

#future = m.make_future_dataframe(periods=14)
#future.tail()

forecast = m.predict(df_test)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#m.plot(forecast)

#draw surge multiplier and mean ETA for every 3 min period
plt.plot(forecast['ds'],forecast['yhat'],color='green')
plt.plot(forecast['ds'],df_new['y'].iloc[10000:], color='blue')

plt.show()

#m.plot_components(forecast)
