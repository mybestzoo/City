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

# filter data by location id
df_loc1 = df[df['locationID']==1]

# drop columns id, timestamp, locationID
df_loc1 = df_loc1.drop(['id','timestamp','locationID'],axis=1)

# draw boxplot of surge_multiplier by expected_wait_time
df_loc1.boxplot(column="surge_multiplier",by="expected_wait_time")

# group by expected_wait_time and calculate median surge_multiplier
group = df_loc1.groupby(['expected_wait_time']).median()

# group by expected_wait_time and calculate mode of surge_multiplier
group = df_loc1.groupby(['expected_wait_time']).agg({'surge_multiplier': lambda x:stats.mode(x)[0]})

# plot mean/median/mode surge_multiplier by expected_wait_time
plt.plot(group)

# ***************************************************************************

# load data again and filter data by location id
df_loc1 = df[df['locationID']==1]

# plot surge_multiplier over time
plt.plot(df_loc1['timestamp'].head(200),df_loc1['surge_multiplier'].head(200))

# plot ETA over time
plt.plot(df_loc1['timestamp'].head(200),df_loc1['expected_wait_time'].head(200)/np.max(df_loc1['expected_wait_time'].head(200)))

# delete first rowas to adjust to surge mulitplier update period (3 min = 12 rows)
df_new = df_loc1.drop(df.index[[0,7,12,17,21,28,31,36,40]])

# make timestamp an index
df_new.index = pd.to_datetime(df_new['timestamp'])

# resample ETA and multiplier in 3min intervals calculating mean value inside each interval
y1 = df_new.expected_wait_time.resample('3min', how='mean') # resampled ETA
y2 = df_new.surge_multiplier.resample('3min', how='mean') # resampled surge multiplier

# add a column with mean expected wait time y2 to the dataframe
wait_time_mean = []
i = -1
j = -1
for index, row in df_new.iterrows():
    i += 1
    if i%12==0:
        j += 1
    wait_time_mean.append(y1[j])    
    
df_new['wait_time_mean'] = wait_time_mean

#draw surge multiplier and mean ETA for every 3 min period
plt.plot(df_new['timestamp'].head(200),df_new['surge_multiplier'].head(200))
plt.plot(df_new['timestamp'].head(200),df_new['wait_time_mean'].head(200)/np.max(df_new['wait_time_mean'].head(200)))

