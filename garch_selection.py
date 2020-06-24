#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GARCH Model Selection
    Created: Jun 2020
    @author: talespadilha
"""
import pandas as pd
import general_functions as gf
from itertools import product
from arch import arch_model
from arch.univariate import ConstantMean, EGARCH

def garch_class(series: pd.Series, max_lag: int, pwr: int):
    """Selects the model with the most appropriate number of lags for models of the TARCH and GJR-GARCH class according to BIC.
    
    Args:
        series: time series we want to analyse.
        max_lag: maximum number og lags to be considered in the models.
        power: power of the GARCH model.
    
    Returns:
        selected_model: dict with tuple (p, o, q) and BIC of the selected model
    """
    bics = {}
    for model in product(range(max_lag+1), repeat=3):
        if model[0]==0 and model[1]==0:
            continue
        mod = arch_model(series, p=model[0], o=model[1], q=model[2], power=pwr)
        bics[model] = mod.fit(disp="off").bic
    # Getting the model that minimizes BIC:
    min_key = min(bics, key=bics.get)
    selected_model = {"order": min_key, "bic": bics[min_key]}
    return selected_model

def egarch_class(series: pd.Series, max_lag: int):
    """Selects the model with the most appropriate number of lags for models of the EGARCH class according to BIC
    
    Args:
        series: time series we want to analyse.
        max_lag: maximum number of lags to be considered in the models.
    
    Returns:
        selected_model: dict with tuple (p, o, q) and BIC of the selected model.
    """
    bics = {}
    for model in product(range(max_lag+1), repeat=3):
        if model[0]==0 and model[1]==0:
            continue
        # Setting the volatility and mean models:
        vol_mod = EGARCH(p=model[0], o=model[1], q=model[2])
        mod = ConstantMean(series, volatility=vol_mod)
        bics[model] = mod.fit(disp="off").bic
    # Getting the model that minimizes BIC:
    min_key = min(bics, key=bics.get)
    selected_model = {"order": min_key, "bic": bics[min_key]}
    return selected_model

def garch_select(series: pd.Series, max_lag=2, outlier=None):
    """Selects the best GARCH model for a mean zero within the TACH, GJR-GARCH, and EGARCH classes for a mean zero series.
    
    Args:
        series: time series we want to analyse.
        max_lag: maximum number og lags to be considered in the models.
    
    Returns:
        selected_model: dict with model class, tuple (p, o, q) and BIC of the selected model.
    """
    # Removing outliers
    if outlier!=None:
        series = gf.remove_outliers(series, outlier)
    # Getting the best model for each class
    tarch = garch_class(series, max_lag, 1)
    gjr = garch_class(series, max_lag, 2)
    egarch = egarch_class(series, max_lag)
    # Selecting the best order and BIC of each class
    models_order = {"tarch":tarch["order"], "gjr":gjr["order"],\
                    "egarch":egarch["order"]}
    models_bic = {"tarch":tarch["bic"], "gjr":gjr["bic"], "egarch":egarch["bic"]}
    # Selecting the best model amogst them all
    min_key = min(models_bic, key=models_bic.get)
    bic_model = {"model": min_key, "order": models_order[min_key], \
                      "bic": models_bic[min_key]}
    return bic_model

def garch_volatility(rets: pd.DataFrame, out=None):
        """Selects the best garch model and returns estimated volatility.
        
        Args:
            rets: Series of demeaned returns
            outlier: int number of z-scores to remove outliers (default is no outlier removal).
        
        Returns:
            vol: Series with the fitted conditional volatility.
            model: dict with the model used to estimate conditional volatility.
        """
        # Getting the model, estimating and getting conditional volatility
        model = garch_select(rets, outlier=out)
        if model['model']== 'tarch': 
            mod = arch_model(rets, p=model['order'][0], o=model['order'][1],
                             q=model['order'][2], power=1)
            vol = mod.fit(disp="off").conditional_volatility
        elif model['model']=='gjr':
            mod = arch_model(rets, p=model['order'][0], o=model['order'][1],
                             q=model['order'][2])     
            vol = mod.fit(disp="off").conditional_volatility
        elif model['model']=='egarch':
            vol_mod = EGARCH(p=model['order'][0], o=model['order'][1], 
                             q=model['order'][2])
            mod = ConstantMean(rets, volatility=vol_mod)
            vol = mod.fit(disp="off").conditional_volatility
        else:
            raise NameError('model type not defined')
        
        return vol, model
    
if __name__=="__main__":
    print("This file contains the univariate GARCH model selection functions.")
    print("Only use file to import functions!")
