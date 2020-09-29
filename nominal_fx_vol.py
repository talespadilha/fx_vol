#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study of nominal exchange rate volatility
    Created: Nov 2019
    @author: talespadilha
"""
import os
os.chdir('/Users/talespadilha/Documents/Projects/fx_vol')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import general_functions as gf
from arch import arch_model
from nominal_fx_data import nominal_import
from garch_selection import garch_volatility

import statsmodels.api as sm


def est_ols_ar(series, factors):
    y = series[~series.isna()]
    factors = factors[~series.isna()]
    x = pd.concat([factors.rename('f'), y.shift(1).rename('lag')], axis=1)
    x = sm.add_constant(x)
    mod = sm.OLS(y[1:], x[1:])
    res = mod.fit()

    return res

def est_ols(series, factors):
    x = sm.add_constant(factors.rename('f'))
    y = series[~series.isna()]
    x = x[~series.isna()]
    mod = sm.OLS(y, x)
    res = mod.fit()

    return res


def corr_list(df):
    l=[]
    for i in range(len(df)):
        for j in range(i):
            if i>j:
                l.append(df.iloc[i,j])
    ls = pd.Series(l)
                
    return ls

def garch11(series):
    mod = arch_model(series, p=1, o=0, q=1)
    vol = mod.fit(disp="off").conditional_volatility
    
    return vol


def calc_rv(df, freq='MS'):
    squared=df**2
    final_df = squared.resample(freq).mean()
    
    return final_df
                

path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'


#%% Importing Data

fx = nominal_import('2000-01-01', '2020-06-30', path)
# Adding CLP
clp = pd.read_csv(path+'clp.csv', index_col=0, parse_dates=True)
ids = fx.index[~fx['EUR'].apply(np.isnan)]
fx['CLP'] = clp.loc[ids, :]
fx = fx.reindex(sorted(fx.columns), axis=1)


#%%

# Getting returns and getting rid of extreme values
r_fx = (fx.pct_change()*100).dropna(how='all')
r_fx[r_fx>10] = 10
r_fx[r_fx<-10] = -10

EMS, DMS = gf.markets_set(r_fx, path)

#%% Taking a look jointly
# Selecting the set of countries
countries = EMS + DMS

f = r_fx[countries].mean(axis=1)

# Estimating the mean factor model
    # First checking the AR behaviour:
t = {}
for country in countries:
    res = est_ols_ar(r_fx[country], f)
    t[country] = res.tvalues['lag']
tstats = pd.Series(t)

e={}
b={}
for country in countries:
    if abs(tstats[country])>1.96:       
        res = est_ols_ar(r_fx[country], f)
    else:
        res = est_ols(r_fx[country], f)
    e[country] = res.resid
    b[country] = res.params
e_hat = pd.concat(e, axis=1)
betas = pd.concat(b, axis=1)


print(f"Median factor effect {betas.loc['f'].median()}.")
betas.loc['f'].hist()
e_corr = e_hat.corr()
corr_list(r_fx[countries].corr()).hist()
print(f"Mean Return Corr: {corr_list(r_fx[countries].corr()).mean()}")
print(f"Median Return Corr: {corr_list(r_fx[countries].corr()).median()}")
corr_list(e_corr).hist()
print(f"Mean e Corr: {corr_list(e_corr).mean()}")
print(f"Median e Corr: {corr_list(e_corr).median()}")

e2_corr = (e_hat**2).corr()
corr_list(e2_corr).hist()
print(f"Mean e2 Corr: {corr_list(e2_corr).mean()}")
print(f"Median e2 Corr: {corr_list(e2_corr).median()}")


e_hat10 = e_hat*10
v = {}
m = {}
for country in e_hat10:
    v[country] = garch11(e_hat10[country].dropna())  
vol = pd.concat(v, axis=1)/10


std_e = e_hat/vol
se2 = std_e**2
se2_corr = (std_e**2).corr()
corr_list(se2_corr).hist()
print(f"Mean e2 Corr: {corr_list(se2_corr).mean()}")
print(f"Median e2 Corr: {corr_list(se2_corr).median()}")

corr_list(std_e.resample('MS').mean().corr()).hist()
corr_list(std_e.resample('MS').mean().corr()).mean()

corr_list(calc_rv(std_e).corr()).hist()
corr_list(calc_rv(std_e).corr()).mean()

psi = se2 - 1


# Testing for GEOVOL
N = psi.shape[1]
T = psi.shape[0]

xi = np.sqrt((N*T)/((N-1)/2))*np.sum(corr_list(se2_corr))

a=[]
for i in range(N):
    for j in range(i):
        if i>j:
            for t in range(T):
                a.append(psi.iloc[t,i]*psi.iloc[t,j])
a_series = pd.Series(a)

b=[]
for i in range(N):
    for t in range(T):
        b.append(psi.iloc[t,i]**2)
b_series = pd.Series(b)

xi = np.sqrt((N*T)/((N-1)/2)) * (a_series.sum()/b_series.sum())
xi

#%% Importing GEOVOL data
x_daily = pd.read_csv(path+'geovols/daily_dm.csv', index_col=0)
x_daily.index = pd.to_datetime(x_daily.index, format='%d/%m/%Y')
s = pd.read_csv(path+'geovols/s_dm.csv', index_col=0)
x_monthly = x_daily.resample('MS').mean()

g = {}
ccs = list(std_e.columns)
#ccs.remove('ILS')
for country in ccs:
    g[country] = (s.loc[country].values * x_daily) + 1 - s.loc[country].values
g = pd.concat(g, axis=1)
g.columns = g.columns.droplevel(1)


corr_list(se2.corr()).hist()
print(f"Mean e2 Corr: {corr_list(se2.corr()).mean()}")

epsilon = se2 / g
corr_list(epsilon.corr()).hist()
print(f"Mean epsilon Corr: {corr_list(epsilon.corr()).mean()}")


# Checking monthly
g_monthly = {}
for country in ccs:
    g_monthly[country] = (s.loc[country].values * x_monthly) + 1 - s.loc[country].values
g_monthly = pd.concat(g_monthly, axis=1)
g_monthly.columns = g_monthly.columns.droplevel(1)


se2_monthly = (std_e**2).resample('MS').mean()
corr_list(se2_monthly.corr()).hist()
print(f"Mean e2 Corr: {corr_list(se2_monthly.corr()).mean()}")

epsilon_m = se2_monthly/g_monthly
corr_list(epsilon_m.corr()).hist()
print(f"Mean epsilon Corr: {corr_list(epsilon_m.corr()).mean()}")


#%% Testing for geovol
psi = epsilon - 1


# Testing for GEOVOL
N = psi.shape[1]
T = psi.shape[0]

a=[]
for i in range(N):
    for j in range(i):
        if i>j:
            for t in range(T):
                a.append(psi.iloc[t,i]*psi.iloc[t,j])
a_series = pd.Series(a)

b=[]
for i in range(N):
    for t in range(T):
        b.append(psi.iloc[t,i]**2)
b_series = pd.Series(b)

xi = np.sqrt((N*T)/((N-1)/2)) * (a_series.sum()/b_series.sum())
xi



