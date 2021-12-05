
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/jesusalvarezc/FInal_Project                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


import statsmodels.api as sm
from scipy import stats
from pylab import rcParams
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def validacion_operacion(timestamp, tipo, volumen, precio_ini, precio_fin, takeprofit, stoploss):
    if tipo == 'Venta':
        tp = precio_ini - takeprofit
        sl = precio_ini + stoploss
    else:
        tp = precio_ini + takeprofit
        sl = precio_ini - stoploss

    tabla = pd.DataFrame(
        {'Concepto': ['DateTime', 'Posicion', 'Volumen', 'Precio inicial', 'Precio final',
                      'Takeprofit', 'Stoploss'],
         'Valor': [timestamp, tipo, volumen, precio_ini, precio_fin, tp, sl]
         }
    )
    return tabla


# Statistical

def serie_tiempo(data):
    fig = px.line(data, x='DateTime', y='Actual')
    return fig.show()


# A y AP
def autocorr(data):
    autocorrelacion = plot_acf(data.Actual)
    autocorrelacion_parcial = plot_pacf(data.Actual)
    return autocorrelacion, autocorrelacion_parcial


# prueba heterocedasticidad
def hetero(data):
    fig, ax = plt.subplots(1, 1)
    ans = sm.qqplot(data.Actual, line='q', fit=True, ax=ax)


# Normalidad
def normalidad(data):
    print('Kursotis:', stats.kurtosis(data.Actual))
    print('Skewness:', stats.skew(data.Actual))
    print('Shapiro-Wilk: ', stats.shapiro(data.Actual))
    print('D Agostino:', stats.normaltest(data.Actual))


def estacionalidad(data):
    rcParams['figure.figsize'] = 16, 6
    descomp = sm.tsa.seasonal_decompose(data.Actual, model='additive', period=12)
    descomp.plot()
    # DICKEY-FULLER
    data_test = adfuller(data.Actual)
    if data_test[1] > 0.05:
        ans = 'No hay estacionariedad'
    else:
        ans = 'Hay estacionariedad'

    return print((ans, data_test[1])), plt.show()


# Datos Atípicos
def atipicos(data):
    ax = sns.boxplot(x='Actual', data=data)


# Computational

def clasificacion(data):
    res = []
    for i in range(len(data)):
        actual = data.iloc[i,1]
        consensus = data.iloc[i,2]
        previous = data.iloc[i,3]
        if actual >= consensus >= previous:
            res.append("A")
        elif actual >= consensus < previous:
            res.append("B")
        elif actual < consensus >= previous:
            res.append("C")
        elif actual < consensus < previous:
            res.append("D")
    data["Escenario"] = res
    return data


def metrics(data: dict, indx: pd.DataFrame):
    res_metric = indx[["DateTime", "Escenario"]]
    directions = []
    pips_alctas = []
    pips_bajtas = []
    volatilidad = []
    for i in list(data.keys()):
        df = data[i]
        o0 = df.loc[0]["open"]
        direction = df.iloc[-1, :]["close"] - df.loc[0]["open"]
        directions.append(direction)
        alctas = [high - o0 for high in df.loc[df.index >= 0, "high"] if high - o0 > 0]
        alctas_sum = int(np.sum(alctas) * 1000)
        pips_alctas.append(alctas_sum)
        bajistas = [o0 - low for low in df.loc[df.index >= 0, "low"] if o0 - low > 0]
        bajistas_sum = int(np.sum(bajistas) * 1000)
        pips_bajtas.append(bajistas_sum)
        vol = [df.high.iloc[v] - df.low.iloc[v] for v in range(len(df))]
        vola = int(np.sum(vol) * 1000)
        volatilidad.append(vola)
    directions_append = [1 if i > 0 else -1 for i in directions]
    res_metric["direccion"] = directions_append
    res_metric["pip_alcistas"] = pips_alctas
    res_metric["pip_bajistas"] = pips_bajtas
    res_metric["volatilidad"] = volatilidad
    return res_metric


def opt_backtest(metrics: pd.DataFrame):
    train_date = dt.datetime(2019, 1, 1)
    test_date = dt.datetime(2019, 2, 1)
    train_metrics = metrics[metrics.Escenario <= train_date]
    train = train_metrics["Escenario"]
    train["operacion"] = ["compra" if i == 1 else "venta" for i in train_metrics["direccion"]]
    train["tp"] = [i for i in train_metrics[""]]

    return train

