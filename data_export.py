#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:20:49 2021

@author: talespadilha
"""

import pandas as pd
pd.read_csv(data_path+'dxy.csv')


data = pd.read_excel(data_path+'reer_imf.xlsx', header = [0,1], index_col = [0])
data.index = pd.to_datetime(data.index, format='%b %Y')
data.to_csv(data_path+'reer_levels.csv')
(data.pct_change()*100).to_csv(data_path+'reer_returns.csv')

data_monthly = []