
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/jesusalvarezc/FInal_Project                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import datetime as dt
import numpy as np


def peso_usd(amount):
    return 1/amount


def f_leer_archivos(file):
    data = pd.read_csv("0files/"+file)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    # 24 porque son 2 años
    data = data.iloc[:24, :]
    data = data.sort_values(by='DateTime', ascending=True)
    return data


def f_leer_archivos_varios(*files):
    result = pd.DataFrame()
    for file in files:
        data = pd.read_csv("0files/"+file)
        result = pd.concat([result, data])
    result.close = np.divide(1, result.close)
    result.open = np.divide(1, result.open)
    high1 = np.divide(1, result.low)
    low1 = np.divide(1, result.high)
    result.high = high1
    result.low = low1
    result.timestamp = [dt.datetime.strptime(i, "%d/%m/%y %H:%M") for i in result.timestamp]

    return result


def look_at_change(data_ind: pd.DataFrame, data_change: pd.DataFrame):
    res = {}
    for i in range(len(data_ind)):
        dic_ind = data_ind.iloc[i, 0]
        delta = dt.timedelta(minutes=30)
        start = dic_ind - delta
        end = dic_ind + delta
        res_to_append = data_change[(data_change["timestamp"] <= end) & (data_change["timestamp"] >= start)]
        res[dic_ind] = res_to_append

    return res
