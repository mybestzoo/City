# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read data
df = pd.read_csv("uber_DM_3h.csv")

df_uberX = df[df.product_type == 'uberKIDS']
df_uberX = df_uberX.reset_index()

# plot surge_multiplier over time
plt.plot(df_uberX['surge_multiplier'])
