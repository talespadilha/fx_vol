#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study of real exchange rate volatility
    Created: Nov 2019
    @author: talespadilha
"""

# Setting work directory
import os
os.chdir('/Users/talespadilha/Documents/Projects/fx_vol')

# Standard Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Idiosyncratic Libraries
import general_functions as gf
from real_fx_data import real_import
from real_fx_data_old import real_import as real_import_old
from garch_selection import garch_volatility
from linear_models import cs_ardl, nw_ols, cs_ardl_dxy


#%% Importing Data
data_path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'
df = real_import(data_path, 'monthly_data.xlsx', 'us_data.xlsx', base_fx='USD') 
#df = real_import_old(data_path, 'monthly_data.xlsx', 'us_data.xlsx') 


# Adjusting for 21st century data
#df= df.loc[pd.to_datetime('01-2000', format='%m-%Y'):, :]

#%% Real FX Vol
# Creating a DataFrame with real exchange rate for each country
real_fx = df.loc[:, (slice(None), 'r')]

# Getting the returns of real exchange rate
real_rets = real_fx.pct_change()*100
real_rets.columns = real_rets.columns.droplevel(-1)
# Standarizing 'real_rets'
norm_real_rets = (real_rets)/real_rets.std()

# Studying volatility
real_vol = {}
model = {}
for currcy in real_rets:
    real_vol[currcy], model[currcy] = garch_volatility(norm_real_rets[currcy].dropna(), out=5)
# Rescalling and annualizing volatility?
real_vol = pd.concat(real_vol, axis=1)
#r_vol = real_vol*real_rets.std()*np.sqrt(12)

#%% Doing the DCC Analysis in MATLAB
df_matlab = df.copy()
df_matlab.columns = df_matlab.columns.map('_'.join)
df_matlab.index = df_matlab.index.strftime("%Y%m%d")
(df_matlab.pct_change()*100).to_csv(data_path+'df_rets.csv')

# Run m-file

# Importing
dcc = pd.read_csv(data_path+'dcc.csv', header=[0], index_col=[0]).iloc[1:]
# Formating index and columns back to pythonic style
dcc.index.name = None
dcc.index = real_vol.index
dcc.columns = df.columns
d = dict(zip(dcc.columns.levels[1], ['var_e','var_p','cov']))
dcc = dcc.rename(columns=d, level=1)

#%% Importing GEOVOL
#geovol_d = pd.read_csv(data_path+"m_xhat.csv", header=0, names=["month","geovol"], 
#                     index_col="month", parse_dates=True)
#geovol = pd.read_csv(data_path+"xhat_df_real_and_nominal.csv", index_col="Date", 
#                     parse_dates=True) 

#ids = pd.Series(geovol_d.index)
#geovol_d.index = ids.apply(lambda dt: dt.replace(day=1))

#%% Fixing For GEOVOL Date
#fist_month = geovol_d.index[0]
#last_month = geovol_d.index[-1]
#real_vol = real_vol.loc[fist_month:last_month, :]
#dcc = dcc.loc[fist_month:last_month, :]

#%% Simple Decompostion
# Adding a column level to the 'real_vol' DataFrame
real_vol.columns = pd.MultiIndex.from_product([real_vol.columns, ['var_r']])

# Estimating the dependencies 
betas = {}
for currcy in real_rets:
    c_df = pd.concat([real_vol[currcy], dcc.xs(currcy, axis=1)], axis=1).dropna()
    c_df = gf.df_outliers(c_df, 3)
    betas[currcy],_ = nw_ols(c_df['var_r'], c_df[['var_e','var_p','cov']])
betas = pd.concat(betas, axis=1)

# Selecting ems and dms
ems, dms = gf.markets_set(df, data_path, base_fx='USD')

# Mean for each group
b_ems = betas[ems].mean(axis=1)
b_dms = betas[dms].mean(axis=1)
print("EMs")
print(b_ems)
print("DMs")
print(b_dms)

#%% Panel Data Analysis

# Load DXY
dxy = pd.read_csv(data_path+"dxy_levels.csv", header=[0], index_col=[0])
dxy.index = pd.to_datetime(dxy.index, format="%d/%m/%Y")
dxy_m = dxy.resample('MS').mean().pct_change()*100
dxy_std = dxy_m/dxy_m.std()
dxy_vol, dxy_model = garch_volatility(dxy_std.dropna(), out=5)
dxy_vol.name = 'DXY Vol'

# EMs
em_params, em_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, ems)
em_mg = gf.trim_mean(em_params, trim_param=0.01)
em_p = gf.trim_mean(em_pvalues, trim_param=0.01)

# DMs
dm_params, dm_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, dms)
dm_mg = gf.trim_mean(dm_params, trim_param=0.01)
dm_p = gf.trim_mean(dm_pvalues, trim_param=0.01)

print("EMs Params")
print(em_mg)
print("DMs Params")
print(dm_mg)

print("EMs P")
print(em_p)
print("DMs P")
print(dm_p)

# All countries
all_c = ems+dms
all_params, all_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, all_c)
all_mg = gf.trim_mean(all_params, trim_param=0.01)
all_p = gf.trim_mean(all_pvalues, trim_param=0.01)
print("All Coeff")
print(all_mg)
print("All Ps")
print(all_p)





