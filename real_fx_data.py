#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Data to Real FX Analysis
    Created: Nov 2019
    @author: talespadilha
"""

import pandas as pd
import numpy as np

def imf_import(data_path: str, file_name: str):
    """Import data from IMF's (transformed) XLSX Excel file.
    
    Args:
        data_path: str with the path for where the XLSX file is located.
        file_name: str with the name of the file we want to import.
   
    Returns:
        data: DataFrame with the imported fx and cpi series for each country
    """
    #Importing the data:
    data = pd.read_excel(data_path+file_name, header = [0,1], index_col = [0])
    data.index = pd.to_datetime(data.index, format='%b %Y')

    return data


def create_df_logs(data: pd.DataFrame, us_prices: pd.DataFrame): 
    """ Creates the data frame with real exchange rate, nominal exchange rate, and price differentials (to US pices).
    
    Args:
        data: DataFrame with the nominal exchange rates and price levels for all countries being considered.
        us_prices: DataFrame with price levels for the US.
        
    Returns:
        df: DataFrame with the information for each country relative to the US.
    """
    # Setting variables to logs:
    data = np.log(data)
    us_prices = np.log(us_prices)
    # Building the data frame with the info we want:
    cols =  pd.MultiIndex.from_product([list(data.columns.levels[0]),
                                        ['r', 'e', 'p']],
                                        names=['country', 'variable'])
    df = pd.DataFrame(index = data.index, columns = cols)
    # Filling this data frame:
    df.loc[:, (slice(None), 'e')] = data.xs('fx', axis=1, level=1).values
    df.loc[:, (slice(None), 'p')] = (data.loc[:, (slice(None), 'cpi')] -
                                        us_prices.values).values
    df.loc[:, (slice(None), 'r')] = (df.xs('e', axis=1, level= 1) - 
                                       df.xs('p', axis=1, level= 1).values).values
   
    return df

def create_df_levels(data: pd.DataFrame, us_prices: pd.DataFrame): 
    """ Creates the data frame with real exchange rate, nominal exchange rate, and price ratios (to US pices).
    
    Args:
        data: DataFrame with the nominal exchange rates and price levels for all countries being considered.
        us_prices: DataFrame with price levels for the US.
        
    Returns:
        df: DataFrame with the information for each country relative to the US.
    """
    # Building the data frame with the info we want:
    cols =  pd.MultiIndex.from_product([list(data.columns.levels[0]),
                                        ['r', 'e', 'p']],
                                        names=['country', 'variable'])
    df = pd.DataFrame(index = data.index, columns = cols)
    # Sorting 'data' dataframe:
    data = data.sort_index(axis=1)
    # Filling this data frame:
    df.loc[:, (slice(None), 'e')] = data.xs('fx', axis=1, level=1).values
    df.loc[:, (slice(None), 'p')] = (us_prices.values/
                                     data.loc[:, (slice(None), 'cpi')]).values
    df.loc[:, (slice(None), 'r')] = (df.xs('e', axis=1, level= 1)* 
                                     df.xs('p', axis=1, level= 1).values).values
   
    return df
    

def real_import(data_path: str, data_file: str, us_file: str):
    """ Imports different datasets and perform the transformations required for the study.
    
    Args:
        data_path: str with the location of the data files.
        data_file: str with the name of the file with fx and cpi info.
        us_file: str with the name of the file with US cpi info.
    
    Returns:
        final_df: DataFrame with the following series for each country:
            e: log of nominal exchange rate.
            p: price differential to the US (log(CPI_i)-log(CPI_US)).
            r: real exchange rate (e-p).
        df: DataFrame with original cpi and fx info for each country.
    """
    # Importing yearly data:
    data = imf_import(data_path, data_file) 
    us_data = imf_import(data_path, us_file)
    # Normalizing prices to most recent observation
    data.loc[:, (slice(None), 'cpi')] = data.loc[:, (slice(None), 'cpi')] \
        / data.loc[data.index[-1], (slice(None), 'cpi')]
    us_data = us_data/us_data.iloc[-1]
    # Creating yearly data data frame:
    final_df = create_df_levels(data, us_data)
    
    return final_df


if __name__ == "__main__":
    print("This file contains the data import functions for FX study.")
    print("Only use file to import functions!")
    