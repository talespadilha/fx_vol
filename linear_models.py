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
    res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    params = res.params
    p_values = res.pvalues

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
    means_real = gf.group_mean(r.loc[:, mkts], level_var=1)
    means_dcc = gf.group_mean(cov.reindex(mkts, axis=1, level='country'), level_var=1)
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
    l1_real = gf.lag_df(r, lag=1, level_var=1)
    #l1_dcc = gf.lag_df(cov, lag=1, level_var=1)
    # Calculating group means
    means_real_all = gf.group_mean(r.reindex(mkts, axis=1, level='country'), level_var=2)
    means_real_usd = gf.group_mean(r.reindex(['USD'], axis=1, level='base'), level_var=0)
    means_real_gbp = gf.group_mean(r.reindex(['GBP'], axis=1, level='base'), level_var=0)
    means_real_jpy = gf.group_mean(r.reindex(['JPY'], axis=1, level='base'), level_var=0)
    means_dcc = gf.group_mean(cov.reindex(mkts, axis=1, level='country'), level_var=2)
    # Getting the lag of the means real series
    l1_means_real = means_real_all.shift(1)
    l1_means_real.columns = ['l1_mean_r']
    # Doing the regressions
    params = {}
    p_values = {}
    for currcy in mkts:
        # TODO: change way to reindex data for xs or reindex
        c_df = pd.concat([r[currcy],
                          l1_real[currcy],
                          cov.xs(currcy, axis=1),
                          l1_means_real,
                          means_dcc,
                          means_real_usd, 
                          means_real_gbp,
                          means_real_jpy,
                          means_real_all
                          #geovol
                          ], axis=1).dropna()
        #c_df = gf.df_outliers(c_df, 3)
        y = c_df['var_r']
        x = sm.add_constant(c_df.loc[:, c_df.columns!='var_r'])
        params[currcy], p_values[currcy] = nw_ols(y, x)
    params = pd.concat(params, axis=1)
    p_values = pd.concat(p_values, axis=1)

    return params, p_values
