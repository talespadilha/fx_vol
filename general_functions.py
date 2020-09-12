#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support Functions for FX Vol Analysis
    Created: Jun 2020
    @author: talespadilha
"""

import math
import numpy as np
import pandas as pd
import scipy.stats as stats

def text_import(file_path: str):
    """Imports a txt file as a list."""
    with open(file_path, 'r') as f:
        txt_list = [line.strip() for line in f]
        
    return txt_list


def remove_outliers(ts: pd.Series, z_number: int):
    """Removes outliers according to Z scores"""
    abs_z_scores = np.abs(stats.zscore(ts))
    new_ts = ts[abs_z_scores<z_number] 
    
    return new_ts


def df_outliers(df: pd.DataFrame, z_number: int):
    """Removes rows with where at least one series has an outlier"""
    new_df = df[(np.abs(stats.zscore(df))<z_number).all(axis=1)]
    
    return new_df


def lag_df(df: pd.DataFrame, lag: int, level_var: int):
    """Creates a DataFrame with the 'lag' lags of 'df'"""
    lag_df = df.shift(lag).copy()
    pref = 'l'+str(lag)+'_'
    lag_df.columns.set_levels([pref+col for col in lag_df.columns.levels[level_var]],
                              level=level_var, inplace=True)
    return lag_df


def trim_mean(df: pd.DataFrame, trim_param: float):
    """Calculates the trimmed mean of a DataFrame for each row"""
    cut = math.ceil(df.shape[1]*trim_param)
    means = {}
    for row in df.index:
        values = df.loc[row, :].dropna().sort_values()
        means[row] = values.iloc[cut:-cut].mean()
    final_series = pd.Series(means)
    
    return final_series

        
def group_mean(df: pd.DataFrame, level_var: int, trim_param=0.1,):
    """Calculates the group mean for the DataFrame"""
    means = {}
    for var in df.columns.levels[level_var]:
        means[var] = trim_mean(df.xs(var, axis=1, level=level_var), trim_param=0.1)
    final_df = pd.concat(means, axis=1)
    final_df.columns = ['mean_'+col for col in final_df.columns]
    
    return final_df

def markets_set(final_df: pd.DataFrame, data_path: str):
    """ Sets which markets are EMs and DMs"""
    # Getting the lists of EMs and DMs
    ems = text_import(data_path+'ems.txt')
    dms = text_import(data_path+'dms.txt')
    # Checking if there match with the data file:
    if set(final_df.columns.get_level_values(0).unique()) != set(ems+dms):
        raise NameError('Update EMs and DMs lists!')
        
    return ems, dms 
    

if __name__ == "__main__":
    print("This file contains the support functions for FX study.")
    print("Only use file to import functions!")