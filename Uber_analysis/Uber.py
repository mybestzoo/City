# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Read data
df = pd.read_csv("uber_4weeks.csv")

df_loc1 = df[df['locationID']==1]
df_loc1 = df_loc1.drop(['id','timestamp','locationID'],axis=1)

df_loc1.boxplot(column="surge_multiplier",by="expected_wait_time")

group = df_loc1.groupby(['expected_wait_time']).median()

group = df_loc1.groupby(['expected_wait_time']).agg({'surge_multiplier': lambda x:stats.mode(x)[0]})

plt.plot(group)
plt.scatter(df_loc1['expected_wait_time'].head(100000),df_loc1['surge_multiplier'].head(100000))

plt.plot(df_loc1['timestamp'].head(200),df_loc1['surge_multiplier'].head(200))
plt.plot(df_loc1['timestamp'].head(200),df_loc1['expected_wait_time'].head(200)/np.max(df_loc1['expected_wait_time'].head(200)))

df_new = df_loc1.drop(df.index[[0,7,12,17,21,28,31,36,40]])
df_new.index = pd.to_datetime(df_new['timestamp'])


y1 = df_new.expected_wait_time.resample('3min', how='mean')
y2 = df_new.surge_multiplier.resample('3min', how='mean')

plt.plot(y2.head(1000))
plt.plot(y1.head(1000)/max(y1.head(1000)))

wait_time_mean = []
i = -1
j = -1
for index, row in df_new.iterrows():
    i += 1
    if index%12==0:
        j += 1
    wait_time_mean.append(y1[j])    
