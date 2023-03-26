# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:13:38 2023

@author: akhil
"""

import pandas as pd
import numpy as np
import datetime as dt


nifty_min = pd.read_parquet('nifty_minute_data.parquet')

nifty_min['date'] = [i.date() for i in nifty_min.index]

# nifty_min = nifty_min.drop('date', axis=1)

dates = set(nifty_min['date'].tolist())
dates = list(dates)
dates.sort()

nifty_hour = pd.DataFrame(columns=nifty_min.columns[:-1])

temp = pd.DataFrame()
# temp.clear()

for d in dates:
    print(d)
    temp = nifty_min[nifty_min['date']==d].copy()
    temp_resampled = pd.DataFrame(columns = temp.columns[:-1])
    temp_resampled['open'] = temp['open'].resample('60min').first()
    temp_resampled['high'] = temp['high'].resample('60min').max()
    temp_resampled['low'] = temp['low'].resample('60min').min()
    temp_resampled['close'] = temp['close'].resample('60min').last()
    temp_resampled['symbol'] = temp_resampled['symbol'].fillna('NIFTY')
    # temp_resamped = temp.resample('H').sum()
    nifty_hour = nifty_hour.append(temp_resampled)
    
    
nifty_hour.to_parquet('nifty_hour.parquet')
    











