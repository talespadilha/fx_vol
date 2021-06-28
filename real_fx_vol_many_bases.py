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
from linear_models import cs_ardl, nw_ols



def calculate_real_vol(df: pd.DataFrame)->pd.DataFrame:
    """Calculates real fx vol
    Args:
        df: pd.DataFrame with full data;

    Returns:
        real_vol: pd.DataFrame with estimated real exchange rate series for each
        country.
    """
    # Real FX Vol
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
    
    return real_vol
    

def matlab_generate(df: pd.DataFrame, file_name: str):
    """Generates output for MATLAB"""
    df_matlab = df.copy()
    df_matlab.columns = df_matlab.columns.map('_'.join)
    df_matlab.index = df_matlab.index.strftime("%Y%m%d")
    (df_matlab.pct_change()*100).to_csv(data_path+file_name)


def matlab_import(real_vol: pd.DataFrame, df: pd.DataFrame,
                  file_name: str) -> pd.DataFrame:
    """Imports MATLAB output"""
    # Importing
    dcc = pd.read_csv(data_path+file_name, header=[0], index_col=[0]).iloc[1:]
    # Formating index and columns back to pythonic style
    dcc.index.name = None
    dcc.index = real_vol.index
    dcc.columns = df.columns
    d = dict(zip(dcc.columns.levels[1], ['var_e','var_p','cov']))
    dcc = dcc.rename(columns=d, level=1)
    
    return dcc


#%% Importing Data
data_path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'
df_usd = real_import(data_path, 'monthly_data.xlsx', 'us_data.xlsx', base_fx='USD') 
df_gbp = real_import(data_path, 'monthly_data.xlsx', 'us_data.xlsx', base_fx='GBP') 
df_jpy = real_import(data_path, 'monthly_data.xlsx', 'us_data.xlsx', base_fx='JPY') 

# Calculating real exchange rate vol
real_vol_usd = calculate_real_vol(df_usd)
real_vol_gbp = calculate_real_vol(df_gbp)
real_vol_jpy = calculate_real_vol(df_jpy)


#MATLAB Analysis
# Export
matlab_generate(df_usd, file_name='df_rets.csv')
matlab_generate(df_gbp, file_name='df_rets_gbp.csv')
matlab_generate(df_jpy, file_name='df_rets_jpy.csv')
# Import 
dcc_usd = matlab_import(real_vol_usd, df_usd, file_name='dcc.csv')
dcc_gbp = matlab_import(real_vol_gbp, df_gbp, file_name='dcc_gbp.csv')
dcc_jpy = matlab_import(real_vol_jpy, df_jpy, file_name='dcc_jpy.csv')


#%% Panel Data Analysis

# Setting set
base_currency = 'JPY'
df = df_jpy.copy()
real_vol = real_vol_jpy.copy()
dcc = dcc_jpy.copy()

# Adding a column level to the 'real_vol' DataFrame
real_vol.columns = pd.MultiIndex.from_product([real_vol.columns, ['var_r']])

# Selecting ems and dms
ems, dms = gf.markets_set(df, data_path, base_fx=base_currency)


# EMs
em_params, em_pvalues = cs_ardl(real_vol, dcc, ems)
em_mg = gf.trim_mean(em_params, trim_param=0.01)
em_p = gf.trim_mean(em_pvalues, trim_param=0.01)

# DMs
dm_params, dm_pvalues = cs_ardl(real_vol, dcc, dms)
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
all_params, all_pvalues = cs_ardl(real_vol, dcc, all_c)
all_mg = gf.trim_mean(all_params, trim_param=0.01)
all_p = gf.trim_mean(all_pvalues, trim_param=0.01)
print("All Coeff")
print(all_mg)
print("All Ps")
print(all_p)

#%% Expanded panel analysis
ems, dms0 = gf.markets_set(df_usd, data_path, base_fx='USD')
dms = dms0+['USD']
all_cs = dms+ems

# Real Vol
real_vol = pd.concat({
    'USD': real_vol_usd,
    'GBP': real_vol_gbp,
    'JPY': real_vol_jpy
    }, axis=1)
real_vol=pd.concat([real_vol], axis=1, keys=['var_r']).reorder_levels([1,2,0], axis=1)
real_vol.columns.names = ['base', 'country', 'variable']

# DCC
dcc = pd.concat({
    'GBP': dcc_usd,
    'USD': dcc_gbp,
    'JPY': dcc_jpy
    }, axis=1)
dcc.columns.names = ['base', 'country', 'variable']