from Calibration_script import main
import numpy as np
import matplotlib.pyplot as plt
import os
import configparser

def plot(x,y,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x,y)
    plt.show()

def scatter(x,y,xlabel,ylabel,title,cutoff):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x,y)
    plt.axhline(cutoff,label='threshhold',color='red')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join('../data/validation',title))

def scatter_cut(x,y,x_cut,y_cut,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, color='black')
    plt.scatter(x_cut,y_cut,color='grey')
    #plt.show()
    plt.savefig(os.path.join('../data/validation',title))

def cut_outliers(x, y, channel):
    """
    Cuts points that are to far away from the fit
    :param x: np array
    :return: cut data
    """
    # Calculating the slope
    slopes = (y - y[0])/x
    m = np.polyfit(x[:20],y[:20],deg = 1)
    #print("Slopes array:",slopes)

    tolerance = abs(m[0]*x[0] - m[0]*x[-1])*0.01
    tolerance = abs(y[0]-y[-1])*0.01
    print(m[0],m[-1])
    print(m[0]*x[0],m[-1]*x[-1])
    print("Tolerance:",tolerance)

    # Making array same size as data with only False in it
    cut = np.zeros_like(slopes, dtype=bool)
    # Set False to zero in parts where data gradient is close to zero
    cut[:][np.isclose(np.gradient(y), 0, atol=tolerance)] = True
    print(np.gradient(y))
    print(y[cut])
    scatter(x,np.gradient(y),"x values","Gradients",f"{channel} Plot of gradients",tolerance)
    scatter_cut(x[~cut],y[~cut],x[cut],y[cut],"x values","Gradients",f"{channel} Plot cut by gradient criteria")

    if x[~cut].size == 0:
        return x[~cut], y[~cut], x[cut], y[cut]
    else:
        # calculate mean and standard deviation
        mean, std = np.mean(slopes[~cut]), np.std(slopes[~cut])
        print(f'For slopes: mean: {mean}, standard deviation: {std}')
        #cut to  2 sigma
        cut[np.logical_or((slopes >= 2 * std + mean), (slopes <= mean - 2 * std))] = True
        scatter(x,slopes,"x values","slopes",f"{channel} Plot of slopes",mean+2*std)
        scatter_cut(x[~cut],y[~cut],x[cut],y[cut],"x values","Gradients",f"{channel} Plot cut by standard deviation")

    return x[~cut], y[~cut],x[cut], y[cut]

config = configparser.ConfigParser()
# Getting path from .ini file
config.read("path.ini")

"""
    For Channel 21
"""
channel = 21

path_UvsU = "./../data/Channel_21_U_vs_U.dat"
columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
data_UvsU = main.read_data(path_UvsU, columns_UvsU)

path_IvsI = "./../data/Channel_21_I_vs_I.dat"
columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
data_IvsI = main.read_data(path_IvsI, columns_IvsI)

path_IlimitvsI = "./../data/Channel_21_Ilimit_vs_I.dat"
columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

x_0,y_0,l_0= main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
#x_0, y_0,x_cut_0,y_cut_0 = cut_outliers(x_0, y_0, channel)

"""
    For Channel 0 with scattering values
"""

channel = 0

path_UvsU = "./../data/Channel_0_U_vs_U.dat"
columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
data_UvsU = main.read_data(path_UvsU, columns_UvsU)

path_IvsI = "./../data/Channel_0_I_vs_I.dat"
columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
data_IvsI = main.read_data(path_IvsI, columns_IvsI)

path_IlimitvsI = "./../data/Channel_0_Ilimit_vs_I.dat"
columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

x_1,y_1,l_1 = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
x_1,y_1,x_cut_1,y_cut_1 = cut_outliers(x_1,y_1,channel)

"""
    NOTES
    DISCREPANCY IN PLOTS FOR CHANNEL 0: CUT VALUES DONT MATCH 
    
    1st criteria:
    calculate mean of gradients and cut small gradients
    
    2nd criteria:
    PLOT RESIDUALS AND COMPARE TO STD FOR ANALYSIS
"""