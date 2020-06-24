#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Data from ECB to Nominal FX Analysis
    Created: Jun 2020
    @author: talespadilha
"""

import pandas as pd
import requests
import io
from general_functions import text_import


def ecb_fx_import(currency: str, start_date: str, end_date: str):
    """Imports exchange rate series againt the EUR from ECB API.
    
    Args:
        currency: str with the currency symbol.
        start_date: str with the first date of the series to be imported.
        end_date: str with the last date of the series to be imported.
    
    Returns:
        ts: DataFrame with the dates and series of the 'currency'.
    """
    # Building blocks for the URL
    entrypoint = 'https://sdw-wsrest.ecb.europa.eu/service/'
    resource = 'data'
    flowRef = 'EXR'
    key = 'D.'+currency+'.EUR.SP00.A'
    parameters = {'startPeriod': start_date, 'endPeriod':end_date}
    # Constructing the URL withe building blocks
    request_url = entrypoint + resource + '/'+ flowRef + '/' + key
    # Making the HTTP request
    response = requests.get(request_url, params=parameters, headers={'Accept': 'text/csv'})
    df = pd.read_csv(io.StringIO(response.text))
    ts = df.filter(['TIME_PERIOD', 'OBS_VALUE'], axis=1)
    ts['TIME_PERIOD'] = pd.to_datetime(ts['TIME_PERIOD'])
    ts = ts.set_index('TIME_PERIOD')

    return ts


def nominal_import(start_date:str, end_date:str, data_path:str, base_fx='USD'):
    """Imports nominal FX series from ECBs API and covert them to base currency.
    
    Args:
        start_date: str with the first date of the series to be imported.
        end_date: str with the last date of the series to be imported.
        data_path: str with the location of the text file with the fx codes.
        base_fx: str with the base currency ('USD' set as default).
   
    Returns:
        df_base: DataFrame with daily series for the selected currencies.    
    """
    # Getting the FX codes we will analyse
    fx_codes = text_import(data_path+'/'+'fx_codes.txt')
    # Using a dictionary to get initally store the FX data
    eur_fx = {}
    eur_fx['EUR'] = ecb_fx_import(base_fx, start_date, end_date)
    for fx in fx_codes:
        eur_fx[fx] = ecb_fx_import(fx, start_date, end_date)
    # Creating a DataFrame
    df_eur = pd.concat(eur_fx, axis=1)
    df_eur.columns = df_eur.columns.droplevel(-1)
    df_eur.index = df_eur.index.rename('Date')
    # Transforming to USD
    df_base = df_eur.div(df_eur['EUR'], axis=0)
    df_base['EUR'] = df_base['EUR'].div(df_eur['EUR'], axis=0)

    return df_base


if __name__ == "__main__":
    print("This file contains the data import functions for FX study.")
    print("Only use file to import functions!")
