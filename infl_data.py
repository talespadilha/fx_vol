#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:05:45 2020

@author: talespadilha
"""

# Setting work directory
import os
os.chdir('/Users/talespadilha/Documents/Projects/fx_vol')
data_path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'


from real_fx_data import imf_import

data = imf_import(data_path, 'monthly_data.xlsx')
us = imf_import(data_path, 'us_data.xlsx')
cpi = data.xs('cpi', axis=1, level='variable')
cpi_us = us.xs('cpi', axis=1, level='variable')
infla = cpi.pct_change()*100
infla_us = cpi_us.pct_change()*100
infla_diff = infla - infla_us.values

infla_diff.to_csv(data_path+'pi_diff.csv')