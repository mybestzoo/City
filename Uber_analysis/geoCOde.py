#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 05:37:12 2018

@author: user
"""

import numpy as np
import pandas as pd
import geocoder
g = geocoder.yandex('Басманный, Большая Почтовая улица, дом 61-67, строение 1')
g.geojson

df = pd.read_excel('veloparking.xlsx')

df = df[df['AdmArea'] == 'Центральный административный округ']
df = df.reset_index(drop=True)

df = df[['Address']]
df['lat'] = np.nan
df['lng'] = np.nan

for index, row in df.iterrows():
    address = row['Address']
    g = geocoder.yandex(address)
    try:
        if g.json['state'] == 'Moscow':
            df.set_value(index,'lat',g.json['lat'])
            df.set_value(index,'lng',g.json['lng'])
    except:
        print(g)
        
df = df.dropna()

df.to_csv('veloparking.csv')

df_config = df
df_config['Address'] = df_config.index
df_config.columns = ['location_id', 'latitude', 'longitude']

df_config.to_json(orient='records')
