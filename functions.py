
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/jesusalvarezc/FInal_Project                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Statistical

import statsmodels.api as sm
from scipy import stats
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def linea(data):
    fig = px.line(data, x='DateTime', y='Actual')
    return fig.show()


# A y AP
def autocorr(data):
    A = plot_acf(data.Actual)
    AP = plot_pacf(data.Actual)
    return A, AP


# prueba heterocedasticidad
def hetero(data):
    fig, ax = plt.subplots(1,1, figsize=[12,6])
    a = sm.qqplot(data.Actual, line= 'q', fit  = True, ax = ax)


# Normalidad
def normalidad(data):
    print('Kursotis:', stats.kurtosis(data.Actual))
    print('Skewness:', stats.skew(data.Actual))
    print('Shapiro-Wilk: ', stats.shapiro(data.Actual)) # no más de 50 datos
    print('D Agostino:', stats.normaltest(data.Actual))


def estacion(data):
    # Estacionalidad
    rcParams['figure.figsize'] = 16, 6
    decomposition = sm.tsa.seasonal_decompose(data.Actual, model='additive', period=12)
    decomposition.plot()
    ## Test de DICKEY-FULLER para Estacionariedad
    df_test = adfuller(data.Actual)
    if df_test[1] > 0.05:
        test= print('No hay evidencia de estacionariedad')
    else:
        test= print('Hay evidencia de estacionariedad')

    return print((test,df_test[1])) ,plt.show()


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
