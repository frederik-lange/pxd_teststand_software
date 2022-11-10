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
    scatter(x,np.gradient(y),"$U_{DAC}$","Gradients of $U{out}$",f"Channel {channel}: Plot of gradients",-tolerance)
    scatter_cut(x[~cut],y[~cut],x[cut],y[cut],"$U_{DAC}$","Gradients of $U{out}$",f"Channel {channel}: Plot cut by gradient criteria")

    if x[~cut].size == 0:
        return x[~cut], y[~cut], x[cut], y[cut]
    else:
        # calculate mean and standard deviation
        mean, std = np.mean(slopes[~cut]), np.std(slopes[~cut])
        print(f'For slopes: mean: {mean}, standard deviation: {std}')
        #cut to  2 sigma
        cut[np.logical_or((slopes >= 2 * std + mean), (slopes <= mean - 2 * std))] = True
        scatter(x,slopes,"$U_{DAC}$","Slopes of $U{out}$",f"Channel {channel}: Plot of slopes",mean+2*std)
        scatter_cut(x[~cut],y[~cut],x[cut],y[cut],"$U_{DAC}$","Slopes of $U{out}$",f"Channel {channel}: Plot cut by standard deviation of slopes")
        print("yes")

    return x[~cut], y[~cut],x[cut], y[cut]
def rms_func(y):
    """
    The root definition of standard deviation
    :param y: input array
    :return: rms/std
    """
    mean = np.mean(y)
    N = len(y)
    rms = np.sqrt(np.abs( np.sum(y**2)/N - mean**2 ))
    return rms

def compare(y):
    """ prints rms and std in comparison """
    print(f"Std: {np.std(y)}")
    print(f"Root RMS: {rms_func(y)}")
    return None
def graph_Cleaner(x,y):
    # get x and y values from read in data
    # arrays for points in the vicinity (windows of 15 points)
    x_win, y_win = np.zeros(15, dtype = float), np.zeros(15, dtype = float)
    remove = np.zeros_like(x, dtype = bool)
    # number of data points
    l = len(x)

    # do this for every point
    for k in range(len(x)):
        # loop over windows in x and y
        for i in range(len(x_win)):
            x_win[i], y_win[i] = 0,0
            # for points more than 15 points away from last point
            if(len(x)-k > 15):
                x_win[i], y_win[i] = x[k+i], y[k+i]
            # for points less than 15 points away from the end
            if(len(x)-k <= 15):
                x_win[i], y_win[i] = x[l-15+i], y[l-15+i]
        # sort by size
        x_win, y_win = np.sort(x_win), np.sort(y_win)

        # median of 15 points
        x_med, y_med = np.mean(x_win), np.mean(y_win)
        #rms try
        x_std = rms_func(x_win[2:-2])
        y_std = rms_func(y_win[2:-2])
        # std of 12 values
        # remove points that stray too far
        if( (np.abs((x[k]-x_med)*1.0/x_std) > 5) or (np.abs((y[k]-y_med)*1.0/y_std) > 5) ):
            print(f"Point: {k}, Value: {y[k]}, Median: {y_med}, Std: {y_std}, calculation: {np.abs((y[k]-y_med)*1.0/y_std)}")
            remove[k] = True

    return x[~remove], y[~remove], x[remove], y[remove]

def define_range(x,y):
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    print(f"Maximum and minimum:",ymax,ymin)
    print("Range",range)
    upper_limit = ymin + 0.9*range
    lower_limit = ymin + 0.1*range
    cut1 = y>upper_limit
    cut2 = y<lower_limit
    cut = cut1+cut2
    return x[~cut],y[~cut], x[cut], y[cut]

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
x_0, y_0, x_cut_0gc, y_cut_0gc = graph_Cleaner(x_0,y_0)
x_0, y_0, x_cut_0, y_cut_0 = define_range(x_0,y_0)
y_cut_0 = np.concatenate((y_cut_0,y_cut_0gc), axis=None)
x_cut_0 = np.concatenate((x_cut_0,x_cut_0gc), axis=None)
scatter_cut(x_0,y_0,x_cut_0,y_cut_0, '$U_{DAC}$ [mV]','$U_{out}$ [mV]',f"Channel {channel}: graphCleaner")

"""
    For Channel 0 with scattering values
"""

channel = 5

path_UvsU = f"./../data/Channel_{channel}_U_vs_U.dat"
columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
data_UvsU = main.read_data(path_UvsU, columns_UvsU)

path_IvsI = f"./../data/Channel_{channel}_I_vs_I.dat"
columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
data_IvsI = main.read_data(path_IvsI, columns_IvsI)

path_IlimitvsI = f"./../data/Channel_{channel}_Ilimit_vs_I.dat"
columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

x_1,y_1,l_1 = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
#x_1,y_1,x_cut_1,y_cut_1 = cut_outliers(x_1,y_1,channel)
#scatter_cut(x_1,y_1,x_cut_1,y_cut_1,"","",f"Channel {channel}: Cut outliers")

x_1,y_1,l_1 = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
x_1, y_1, x_cut_1gc, y_cut_1gc = graph_Cleaner(x_1,y_1)
x_1, y_1, x_cut_1, y_cut_1 = define_range(x_1,y_1)
y_cut_1 = np.concatenate((y_cut_1,y_cut_1gc), axis=None)
x_cut_1 = np.concatenate((x_cut_1,x_cut_1gc), axis=None)
#scatter_cut(x_1,y_1,x_cut_1,y_cut_1,'$I_{out(SMU)}$ [mA]','$I_{outMon}$ [mV]',f"Channel {channel}: graphCleaner")

"""
    NOTES
    1st criteria:
    calculate mean of gradients and cut small gradients
    
    2nd criteria:
    PLOT RESIDUALS AND COMPARE TO STD FOR ANALYSIS
"""