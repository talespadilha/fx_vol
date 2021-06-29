#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Models for FX Vol Analysis
    Created: Jun 2020
    @author: talespadilha
"""

import pandas as pd
import numpy as np
import general_functions as gf
import statsmodels.api as sm
from scipy.optimize import minimize


OUTLIERS = [
'JPY_USD',
'CNY_USD',
'CNY_GBP',
'RUB_GBP',
'GBP_JPY',
'AUD_JPY',
'CAD_JPY',
'CHF_JPY',
'SGD_JPY',
'DKK_JPY',
'CNY_JPY',
'RUB_JPY',
]


def get_ols_se(param_values, x, y):
    params = pd.Series(param_values, index=x.columns)
    y_hat = x.mul(params,axis=1).sum(axis=1)
    sse = ((y-y_hat)**2).sum()

    return sse


def nw_ols(y:pd.Series, x:pd.DataFrame):
    """Performs Newey-West heteroscedastic-serial correlation consistent least-squares regression

    Args:
        y: Series of the dependent variab le.
        max_lag: DataFrame of the regressors.

    Returns:
        params: parameters estimated by the model.
        p_values: p-values of the estimated parameters.
    """
    mod = sm.OLS(y, x)
    max_lags = int(x.shape[0]**(1/3))
    nw_res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    p_values = nw_res.pvalues
    #params = nw_res.params
    p0 = nw_res.params
    data_args = (x, y)
    bnds = ((0,1000),)+tuple((-1000,1000) for _ in range(len(p0)-1))
    res = minimize(get_ols_se, p0, args=data_args, bounds=bnds)
    params = pd.Series(res.x, index=p_values.index)

    return params, p_values


def cs_ardl(r:pd.DataFrame, cov:pd.DataFrame, mkts:list):
    """Estimates the cross-sectionally augmented autoregressive distributed lag model for the group defined in 'mkts'

    Args:
        r: DataFrame with the dependent variable.
        cov: DataFrame with the explanatory variables.
        mkt: list with the columns from 'r' that are from the same group.

    Returns:
        params: DataFrame with model parameters for each member of 'mkts'.
        p_values: DataFrame with parameters p-values for each member of 'mkts'.
    """    
    # Creating the lags
    l1_real = gf.lag_df(r, lag=1, level_var=1)
    #l1_dcc = gf.lag_df(cov, lag=1, level_var=1)
    # Calculating group means
    all_ccs = r.columns.get_level_values(0).unique()
    means_real = gf.group_mean(r.loc[:, all_ccs], level_var=1)
    means_dcc = gf.group_mean(cov.reindex(all_ccs, axis=1, level='country'), level_var=1)
    # Getting the lag of the means real series
    l1_means_real = means_real.shift(1)
    l1_means_real.columns = ['l1_mean_r']
    # Doing the regressions
    params = {}
    p_values = {}
    for currcy in mkts:
        c_df = pd.concat([r[currcy],
                          l1_real[currcy],
                          cov.xs(currcy, axis=1),
                          l1_means_real,
                          means_dcc,
                          means_real
                          #geovol
                          ], axis=1).dropna()
        #c_df = gf.df_outliers(c_df, 3)
        y = c_df['var_r']
        x = sm.add_constant(c_df.loc[:, c_df.columns!='var_r'])
        params[currcy], p_values[currcy] = nw_ols(y, x)
    params = pd.concat(params, axis=1)
    p_values = pd.concat(p_values, axis=1)

    return params, p_values


def cs_ardl_two_factors(r:pd.DataFrame, cov:pd.DataFrame, mkts:list):
    """Estimates the cross-sectionally augmented autoregressive distributed lag model for the group defined in 'mkts'

    Args:
        r: DataFrame with the dependent variable.
        cov: DataFrame with the explanatory variables.
        mkt: list with the columns from 'r' that are from the same group.

    Returns:
        params: DataFrame with model parameters for each member of 'mkts'.
        p_values: DataFrame with parameters p-values for each member of 'mkts'.
    """    
    # Creating the lags
    l1_real = gf.lag_df(r, lag=1, level_var=2)
    #l1_dcc = gf.lag_df(cov, lag=1, level_var=1)
    # Calculating group means
    all_ccs = r.columns.get_level_values('country').unique()
    means_real_all = gf.group_mean(r.reindex(all_ccs, axis=1, level='country'), level_var=2)
    means_dcc = gf.group_mean(cov.reindex(all_ccs, axis=1, level='country'), level_var=2)
    # Getting the lag of the means real series
    l1_means_real = means_real_all.shift(1)
    l1_means_real.columns = ['l1_mean_r']
    # Doing the regressions
    params = {}
    p_values = {}
    for bse in r.columns.get_level_values('base').unique():
        non_bse_mkts = [x for x in mkts if x not in [bse]]
        means_real_group = gf.group_mean(r.reindex([bse], axis=1, level='base'), level_var=0)
        means_real_group.columns = ['means_base']
        l1_means_real_group = means_real_group.shift(1)
        l1_means_real_group.columns = ['l1_means_base']
        for currcy in non_bse_mkts:
            c_df = pd.concat([r.xs((bse,currcy), axis=1, level=['base', 'country']),
                              l1_real.xs((bse,currcy), axis=1, level=['base', 'country']),
                              cov.xs((bse,currcy), axis=1, level=['base', 'country']),
                              l1_means_real_group,
                              l1_means_real,
                              means_dcc,
                              means_real_group,
                              means_real_all
                              ], axis=1).dropna()
            #c_df = gf.df_outliers(c_df, 3)
            y = c_df['var_r']
            x = sm.add_constant(c_df.loc[:, c_df.columns!='var_r'])
            if currcy+'_'+bse not in OUTLIERS:
                params[currcy+'_'+bse], p_values[currcy+'_'+bse] = nw_ols(y, x)
    params = pd.concat(params, axis=1)
    p_values = pd.concat(p_values, axis=1)

    return params, p_values
