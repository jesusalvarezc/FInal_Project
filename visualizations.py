
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/jesusalvarezc/FInal_Project                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px


def graf_ind(indicator):
    fechas = indicator["DateTime"]
    fig, ax = plt.subplots()
    ax.plot(fechas, indicator['Actual'], color='tab:purple', label='Actual')
    ax.plot(fechas, indicator['Consensus'], color='tab:green', label='Consensus')
    ax.plot(fechas, indicator['Previous'], color='tab:blue', label='Previous')
    plt.xticks(rotation=45)
    plt.ylabel("Valores del indicador")
    plt.xlabel("Fechas")
    plt.title("Trade balance USA indicator")
    ax.legend(loc='upper right')
    return plt.show()


def graf_val(data1, fecha):
    close = data1['close']
    ax1 = close.plot(color='blue')
    ax1.axvline(fecha, color='red', linestyle='--')
    plt.xticks(rotation=45)
    plt.ylabel("Precios")
    plt.xlabel("Fechas")
    plt.title("USD-MXN")
    return plt.show()


def graf_val_ind(indicator):
    x = indicator["DateTime"]
    y = indicator["Actual", "Consensus", "Previous"]
    plt.plot(x, y)
    plt.xticks(rotation=45)
    plt.ylabel("Valores de Trade balance")
    plt.title("Timestamp")

    return plt.show()
