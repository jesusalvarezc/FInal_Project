
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
