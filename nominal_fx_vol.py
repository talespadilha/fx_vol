#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study of nominal exchange rate volatility
    Created: Nov 2019
    @author: talespadilha
"""

import numpy as np
import pandas as pd
from nominal_fx_data import nominal_import


path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'


fx = nominal_import('2000-01-01', '2020-06-30', path)
clp = pd.read_csv(path+'clp.csv', index_col=0, parse_dates=True)

ids = fx.index[~fx['EUR'].apply(np.isnan)]
fx['CLP'] = clp.loc[ids, :]
fx = fx.reindex(sorted(fx.columns), axis=1)
fx.to_csv('fx.csv')