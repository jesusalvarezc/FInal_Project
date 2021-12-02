
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
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
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
