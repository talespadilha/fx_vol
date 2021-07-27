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
from garch_selection import garch_volatility
from linear_models import cs_ardl, nw_ols, cs_ardl_dxy


def calculate_real_vol(df: pd.DataFrame)->pd.DataFrame:
    """Calculates real fx vol
    Args:
        df: pd.DataFrame with full data;

    Returns:
        real_vol: pd.DataFrame with estimated real exchange rate series for each
        country.
    """
    real_fx = df.loc[:, (slice(None), 'r')]
    real_rets = real_fx.pct_change()*100
    real_rets.columns = real_rets.columns.droplevel(-1)
    norm_real_rets = (real_rets)/real_rets.std()
    real_vol = {}
    for currcy in real_rets:
        real_vol[currcy], _ = garch_volatility(norm_real_rets[currcy].dropna(), out=5)
    real_vol = pd.concat(real_vol, axis=1)

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
    if file_name=='dcc_eur.csv':
        dcc = dcc.dropna(how='all')
    dcc.index = real_vol.index
    dcc.columns = df.columns
    d = dict(zip(dcc.columns.levels[1], ['var_e','var_p','cov']))
    dcc = dcc.rename(columns=d, level=1)

    return dcc


def get_dxy_vol(file_name: str) -> pd.DataFrame:
    dxy = pd.read_csv(data_path+file_name, header=[0], index_col=[0])
    dxy.index = pd.to_datetime(dxy.index, format="%d/%m/%Y")
    dxy_m = dxy.resample('MS').mean().pct_change()*100
    dxy_std = dxy_m/dxy_m.std()
    dxy_vol, dxy_model = garch_volatility(dxy_std.dropna(), out=5)
    dxy_vol.name = 'DXY Vol'

    return dxy_vol

if __name__ == '__main__':
    #%% Importing Data
    data_path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'
    df = real_import(data_path, 'monthly_data.xlsx', 'us_data.xlsx', base_fx='USD')
    real_vol = calculate_real_vol(df)
    matlab_generate(df, file_name='df_rets.csv')
    dcc = matlab_import(real_vol, df, file_name='dcc.csv')
    # Preparing for analysis
    real_vol.columns = pd.MultiIndex.from_product([real_vol.columns, ['var_r']])
    ems, dms = gf.markets_set(df, data_path, base_fx='USD')
    # Model 1
    #EMs
    em_params, em_pvalues, _ = cs_ardl(real_vol, dcc, ems)
    em_mg = gf.trim_mean(em_params, trim_param=0.01)
    em_p = gf.trim_mean(em_pvalues, trim_param=0.01)
    # DMs
    dm_params, dm_pvalues, _ = cs_ardl(real_vol, dcc, dms)
    dm_mg = gf.trim_mean(dm_params, trim_param=0.01)
    dm_p = gf.trim_mean(dm_pvalues, trim_param=0.01)
    print("EMs Params")
    print(round(em_mg, 4))
    print("DMs Params")
    print(round(dm_mg, 4))
    print("EMs P")
    print(round(em_p, 4))
    print("DMs P")
    print(round(dm_p, 4))
    # All countries
    all_c = ems+dms
    all_params, all_pvalues, _ = cs_ardl(real_vol, dcc, all_c)
    all_mg = gf.trim_mean(all_params, trim_param=0.01)
    all_p = gf.trim_mean(all_pvalues, trim_param=0.01)
    print("All Coeff")
    print(round(all_mg, 4))
    print("All Ps")
    print(round(all_p, 4))
    # MODEL 2
    # Load DXY
    dxy_vol = get_dxy_vol("dxy_levels.csv")
    #EMs
    em_params, em_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, ems)
    em_mg = gf.trim_mean(em_params, trim_param=0.01)
    em_p = gf.trim_mean(em_pvalues, trim_param=0.01)
    # DMs
    dm_params, dm_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, dms)
    dm_mg = gf.trim_mean(dm_params, trim_param=0.01)
    dm_p = gf.trim_mean(dm_pvalues, trim_param=0.01)
    print("EMs Params")
    print(round(em_mg, 4))
    print("DMs Params")
    print(round(dm_mg, 4))
    print("EMs P")
    print(round(em_p, 4))
    print("DMs P")
    print(round(dm_p, 4))
    # All countries
    all_c = ems+dms
    all_params, all_pvalues = cs_ardl_dxy(real_vol, dcc, dxy_vol, all_c)
    all_mg = gf.trim_mean(all_params, trim_param=0.01)
    all_p = gf.trim_mean(all_pvalues, trim_param=0.01)
    print("All Coeff")
    print(round(all_mg, 4))
    print("All Ps")
    print(round(all_p, 4))

def chow_analysis():
    """Sample code for Chow test for structural break"""
    real_vol_sample = real_vol.loc['2009-01-01':].copy()
    dcc_sample = dcc.loc['2009-01-01':].copy()
    # All countries
    all_c = ems+dms
    all_params, all_pvalues, all_sse_all = cs_ardl(real_vol, dcc, all_c)
    sse1 = gf.trim_mean(pd.DataFrame(all_sse1).T, trim_param=0.01)[0]
    sse2 = gf.trim_mean(pd.DataFrame(all_sse2).T, trim_param=0.01)[0]
    sse_full = gf.trim_mean(pd.DataFrame(all_sse_all).T, trim_param=0.01)[0]
    n1 = real_vol_sample.shape[0]
    n2 = real_vol_sample.shape[0]
    k = 10
    chow = ((sse_full-(sse1+sse2))/k)/((sse1+sse2)/(n1+n2-2*k))
    print('')