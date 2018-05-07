# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read data
df = pd.read_csv("uberMoscow.csv")

df_uberX = df[df.start_location_id == 1]
df_uberX = df_uberX[df.product_type == 'uberX']
df_uberX = df_uberX.reset_index()
df_uberX.convert_objects(convert_numeric=True)
df_uberX['ETAMean'] = df_uberX['expected_wait_time'].rolling(50).mean()

# plot surge_multiplier over time
plt.plot(df_uberX['surge_multiplier'],color='blue')
plt.plot(df_uberX['expected_wait_time'].convert_objects(convert_numeric=True)/300,color='yellow')
plt.plot(df_uberX['ETAMean']/300,color='green')
plt.show()
