#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study of nominal exchange rate volatility
    Created: Nov 2019
    @author: talespadilha
"""

import pandas as pd
from nominal_fx_data import nominal_import


path = '/Users/talespadilha/Documents/Oxford/Research/Real Exchange Rate Volatility/Data Files/'


fx = nominal_import('2000-01-01', '2019-12-31', path)
