
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


def f_leer_archivos(file):
    data = pd.read_csv("0files/"+file)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    # tomar solo los primeros 24 datos porque se analizan 2 años
    data = data.iloc[:24, :]
    data = data.sort_values(by='DateTime', ascending=True)
    return data


def f_leer_archivos_varios(*files):
    result = pd.DataFrame()
    for file in files:
        data = pd.read_excel("0files/"+file)
        result = pd.concat([result, data])
    return result
