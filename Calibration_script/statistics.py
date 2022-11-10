from Calibration_script import main
from Calibration_script import validation
import numpy as np
import matplotlib.pyplot as plt
import os
import configparser

def outliers(x,y):
    # range criteria
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    print(f"Maximum and minimum:", ymax, ymin)
    print("Range", range)
    upper_limit = ymin + 0.95 * range
    lower_limit = ymin + 0.05 * range
    cut1 = y > upper_limit
    cut2 = y < lower_limit
    cut = cut1 + cut2

    x2,y2 = x[~cut],y[~cut]
    print(len(y2))
    grad = np.gradient(y2)
    print(grad)
    print(np.mean(grad))
    print(np.std(grad))
    remove = np.zeros(len(x2), dtype=bool)
    remove[np.logical_or(grad>np.mean(grad) + np.std(grad) , grad < np.mean(grad) - np.std(grad))] = True
    print(remove)
    print(len(y2),len(remove))
    print(len(y2[~remove]))
    return x2[~remove], y2[~remove], x2[remove], y2[remove]


channel = 1

path_UvsU = f"./../data/Channel_{channel}_U_vs_U.dat"
columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
data_UvsU = main.read_data(path_UvsU, columns_UvsU)

path_IvsI = f"./../data/Channel_{channel}_I_vs_I.dat"
columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
data_IvsI = main.read_data(path_IvsI, columns_IvsI)

path_IlimitvsI = f"./../data/Channel_{channel}_Ilimit_vs_I.dat"
columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

x,y,l = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
x,y,x_cut,y_cut = outliers(x,y)
validation.scatter_cut(x,y,x_cut,y_cut,"x","y",f"Channel {channel}: new")