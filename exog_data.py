#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:05:45 2020

@author: talespadilha
"""

import os
os.chdir('/Users/talespadilha/Documents/Projects/fx_vol')

from real_fx_data import imf_import
import numpy as np

data_path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'

#%% Inflation Differentials

data = imf_import(data_path, 'monthly_data.xlsx')
us = imf_import(data_path, 'us_data.xlsx')
cpi = data.xs('cpi', axis=1, level='variable')
cpi_us = us.xs('cpi', axis=1, level='variable')
infla = cpi.pct_change()*100
infla_us = cpi_us.pct_change()*100

infla_diff = infla - infla_us.values
infla_diff = infla_diff.sort_index(axis=1)
infla_diff.to_csv(data_path+'pi_diff.csv')

infla_12m = ((1+(infla/100)).rolling(12).apply(np.prod, raw=True)-1)*100
infla_us_12m = ((1+(infla_us/100)).rolling(12).apply(np.prod, raw=True)-1)*100
infla_diff_12m = infla_12m - infla_us_12m.values
infla_diff_12m = infla_diff_12m.sort_index(axis=1)
infla_diff_12m.to_csv(data_path+'pi_12m_diff.csv')

#%% Interst Rate Differentials
mmr = imf_import(data_path, 'mmr.xlsx')
mmr_diff = mmr - mmr['USD']
mmr_diff = mmr_diff.droplevel('variable', axis=1)
mmr_diff.pop('USD')
mmr_diff.to_csv(data_path+'i_diff.csv')

mmr_12m = mmr.rolling(12).mean()
mmr_diff_12m = mmr_12m - mmr_12m['USD']
mmr_diff_12m = mmr_diff_12m.droplevel('variable', axis=1)
mmr_diff_12m.pop('USD')
mmr_diff_12m.to_csv(data_path+'i_12m_diff.csv')
