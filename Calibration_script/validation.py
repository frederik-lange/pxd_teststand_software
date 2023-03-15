'''
    Investigation and test of root script and old method for outlier treatment. For the new method, see "statistics.py"
    The plots from this program are deposited in the "validation" folder
'''
from Calibration_script import main
import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
import scipy.optimize as so
import fit

def clear_directory():
    for root, dirs, files in os.walk('../data/validation'):
        for file in files:
            if file.endswith('.png'):
                os.remove(os.path.join(root,file))

def plot(x,y,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x,y)
    plt.show()

def scatter_grad(x,y,xlabel,ylabel,title,cutoff):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x,y)
    if y.mean() > 0:
        cutoff = np.abs(cutoff)
    else:
        cutoff = -np.abs(cutoff)
    plt.axhline(cutoff,label='threshold',color='red')
    plt.axhline(-cutoff, color='red')
    plt.legend()
    #plt.show()
    title = title.replace(" ","_")
    title = title.replace("{","")
    title = title.replace("}", "")
    title = title.replace("$", "")
    plt.savefig(os.path.join('../data/validation',title))
    plt.close()

def scatter_slope(x,y,xlabel,ylabel,title,mean,std):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x,y)
    upper, lower = mean + 2 * std, mean - 2 * std
    plt.axhline(upper,label='threshold',color='red')
    plt.axhline(lower,color='red')
    plt.legend()
    #plt.show()
    title = title.replace(" ","_")
    title = title.replace("{","")
    title = title.replace("}", "")
    title = title.replace("$", "")
    plt.savefig(os.path.join('../data/validation',title))
    plt.close()

def scatter_cut(x,y,x_cut,y_cut,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, color='black')
    plt.scatter(x_cut,y_cut,color='grey')
    #plt.show()
    title = title.replace(" ","_")
    title = title.replace("{","")
    title = title.replace("}", "")
    title = title.replace("$", "")
    plt.savefig(os.path.join('../data/validation',title))
    plt.close()

def cut_outliers(x, y, channel, xlabel, ylabel, title):
    """
    Cuts points that are to far away from the fit
    :param x: np array
    :return: cut data
    """
    # Calculating the slope
    slopes = (y - y[0])/x
    m = np.polyfit(x[:20],y[:20],deg = 1)

    tolerance = abs(m[0]*x[0] - m[0]*x[-1])*0.01
    #tolerance = abs(y[0]-y[-1])*0.01
    #print(m[0],m[-1])
    #print(m[0]*x[0],m[-1]*x[-1])
    #print("Tolerance:",tolerance)

    # Making array same size as data with only False in it
    cut = np.zeros_like(slopes, dtype=bool)
    # Set False to zero in parts where data gradient is close to zero
    cut[:][np.isclose(np.gradient(y), 0, atol=tolerance)] = True
    #print(np.gradient(y))
    #print(y[cut])
    upper,lower = tolerance,-tolerance
    scatter_grad(x,np.gradient(y),xlabel,f"Gradients of {ylabel}",f"Channel {channel} ${title}$ Plot of gradients",tolerance)
    scatter_cut(x[~cut],y[~cut],x[cut],y[cut],xlabel,f"Gradients of {ylabel}",f"Channel {channel} ${title}$ Plot cut by gradient criteria")

    if x[~cut].size == 0:
        return x[~cut], y[~cut], x[cut], y[cut]
    else:
        # calculate mean and standard deviation
        mean, std = np.mean(slopes[~cut]), np.std(slopes[~cut])
        #print(f'For slopes: mean: {mean}, standard deviation: {std}')
        #cut to  2 sigma
        cut[np.logical_or((slopes >= 2 * std + mean), (slopes <= mean - 2 * std))] = True
        scatter_slope(x,slopes,xlabel,f"Slopes of {ylabel}",f"Channel {channel} ${title}$ Plot of slopes",mean,std)
        scatter_cut(x[~cut],y[~cut],x[cut],y[cut],xlabel,f"Slopes of {ylabel}",f"Channel {channel} ${title}$ Plot cut by standard deviation of slopes")

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
    for k in range(l):
        # loop over windows in x and y
        for i in range(len(x_win)):
            #x_win[i], y_win[i] = 0,0
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
        x_med, y_med = np.median(x_win), np.median(y_win)
        #print(x_win[2:-2],y_win[2:-2])
        #rms try
        x_std = rms_func(x_win[2:-2])
        y_std = rms_func(y_win[2:-2])

        #x_std = np.std(x_win[2:-2])
        #y_std = np.std(y_win[2:-2])
        #print(x_med,y_med,x_std,y_std)
        # std of 12 values
        # remove points that stray too far
        #if channel == 5:
            #print(f"Point: {k}, Value: {y[k]}, Median: {y_med}, Std: {y_std}, calculation: {np.abs((y[k] - y_med) * 1.0 / y_std)}")
        if np.abs((x[k]-x_med)*1.0/x_std) > 5 :
            print(f"Point: {k}, Value: {x[k]}, Median: {x_med}, Std: {x_std}, calculation: {np.abs((x[k] - x_med) * 1.0 / x_std)}")
            remove[k] = True
        if np.abs((y[k]-y_med)*1.0/y_std) > 5 :
            print(f"Point: {k}, Value: {y[k]}, Median: {y_med}, Std: {y_std}, calculation: {np.abs((y[k]-y_med)*1.0/y_std)}")
            remove[k] = True
        #print(f"Point: {k}, Value: {y[k]}, Median: {y_med}, Std: {y_std}, calculation: {np.abs((y[k] - y_med) * 1.0 / y_std)}")
    #print(remove)
    return x[~remove], y[~remove], x[remove], y[remove]

def define_range(x,y):
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    upper_limit = ymin + 0.9*range
    lower_limit = ymin + 0.1*range
    cut1 = y>upper_limit
    cut2 = y<lower_limit
    cut = cut1+cut2
    return x[~cut],y[~cut], x[cut], y[cut]

def eval(input,output,x0):
    # assigns one value of an array to the corresponding value of another array
    return output[np.argmin(np.abs(input-x0))]

def bisect(x,y,b):
    print("Bisect")
    range_y = np.abs(np.amax(y) - np.amin(y))
    range_x = np.abs(np.amax(x) - np.amin(x))
    iterations, max_iterations = 0, 50
    deviation = range_y/10
    epsilon = range_y / 50000
    mid_x = np.amin(x) + range_x/2
    right_x = np.amax(x)
    left_x = np.amin(x)
    ycalc = b
    while np.abs(deviation)>epsilon and iterations < max_iterations:
        mid_y = eval(x,y,mid_x)
        #print("mid_x",mid_x, "min",np.min(np.abs(x-mid_x)), f"Mid_y {mid_y}")
        deviation = ycalc - mid_y
        left_y = eval(x,y,left_x)
        right_y = eval(x,y,right_x)
        # right half
        #print(f"ycalc: {ycalc}, ymid {mid_y}, left_y: {left_y}, right_y: {right_y}")
        if ycalc < mid_y and ycalc > right_y:
            #print("Left half")
            left_x = mid_x
            mid_x=mid_x+(right_x-left_x)/2
        # left half
        if ycalc < left_y and ycalc > mid_y:
            #print("right half")
            right_x = mid_x
            mid_x = left_x + (right_x-left_x)/2.0
        iterations += 1
        #print(f"Deviation {deviation}, epsilon {epsilon}")
        #print(f"Iteration {iterations}, ycalc: {ycalc}")
    return mid_x

def root_analysis(x,y):
    print("ROOT analysis:")
    range_y = np.abs(np.amax(y) - np.amin(y))
    range_x = np.abs(np.amax(x) - np.amin(x))
    range_upper = np.amin(y) + 0.95*range_y
    range_lower = np.amin(y) + 0.05*range_y
    high = bisect(x,y,range_upper)
    low = bisect(x,y,range_lower)
    print(f"Lower end: {low}, Upper end: {high}:")
    # apply range:
    mask = (x>high) & (x<low)
    print(np.sum(mask))
    # fit?
    popt, pcov = so.curve_fit(lambda x,a,b: a*x + b, x[~mask],y[~mask])
    print(f"Bisecting: removed {len(x)-np.sum(mask)} points")
    print(popt)
    p0 = (y[-1]-y[0])/(x[-1]-x[0])
    popt_odr,perr_odr,redchi2 = fit.fit_odr(fit_func=lambda m,x,b:m*x+b,x=x[~mask],y=y[~mask],p0=[popt[0],popt[1]])
    print(popt_odr)
    popt[0],popt[1] = popt_odr[0],popt_odr[1]
    y_fit = x * popt[0] + popt[1]
    residuals = y - y_fit
    #print(residuals)
    #remove = np.zeros_like(x)
    remove = np.abs(residuals/y) > 0.02
    #print(remove)
    print("Removed:",np.sum(remove))
    #return x[~remove], y[~remove], x[remove], y[remove], popt[0], popt[1]
    return x[mask],y[mask],x[~mask],y[~mask],popt[0],popt[1], high, low

def validate_outliers():
    path = '../data/example'
    config = configparser.ConfigParser()
    # Getting path from .ini file
    config.read("path.ini")

    for channel in [5,21]: #[0,1,5,10,21]:
        print(channel)
        path_UvsU = os.path.join(path,f'Channel_{channel}_U_vs_U.dat')
        columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5", "unknown 6"]
        data_UvsU = main.read_data(path_UvsU, columns_UvsU)

        path_IvsI = os.path.join(path,f'Channel_{channel}_I_vs_I.dat')
        columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit", "$U_{SMU}$"]
        data_IvsI = main.read_data(path_IvsI, columns_IvsI)

        path_IlimitvsI = os.path.join(path,f'Channel_{channel}_Ilimit_vs_I.dat')
        columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
        data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

        x_0, y_0, l_0 = main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
        x_1, y_1, l_1 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
        x_2, y_2, l_2 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
        x_3, y_3, l_3 = main.get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
        x_4, y_4, l_4 = main.get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')

        #scatter_cut(x_0,y_0,None,None,"$U_{DAC}$","$U_{out}$",f"Channel {channel} DAC Voltage")
        #scatter_cut(x_3,y_3,None,None,"$I_{SMU}$","$I_{outMon}$",f"Channel {channel} ADC I Monitoring")

        x_0, y_0, x_cut_0, y_cut_0 = cut_outliers(x_0, y_0, channel, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]', "U_{DAC}")
        x_1, y_1, x_cut_1, y_cut_1 = cut_outliers(x_1, y_1, channel, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]', "U_{Regulator}")
        x_2, y_2, x_cut_2, y_cut_2 = cut_outliers(x_2, y_2, channel, '$U_{out}$ [mV]', '$U_{load}$ [mV]', "U_{Load}")
        x_3, y_3, x_cut_3, y_cut_3 = cut_outliers(x_3, y_3, channel, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]', "I_{OutMon}")
        x_4, y_4, x_cut_4, y_cut_4 = cut_outliers(x_4, y_4, channel, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]', "I_{Limit}")

        x_0, y_0, l_0 = main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
        x_1, y_1, l_1 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
        x_2, y_2, l_2 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
        x_3, y_3, l_3 = main.get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
        x_4, y_4, l_4 = main.get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')

def compare_chi():
    config_root = configparser.ConfigParser()
    config_root.read('../data/example/constants_from_root.ini')
    path = '../data/example'
    vars = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET', 'ADC_U_REGULATOR_GAIN',
            'ADC_U_REGULATOR_OFFSET',
            'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    for channel in [5,21]:
        path_UvsU = os.path.join(path, f'Channel_{channel}_U_vs_U.dat')
        columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5",
                        "unknown 6"]
        data_UvsU = main.read_data(path_UvsU, columns_UvsU)

        path_IvsI = os.path.join(path, f'Channel_{channel}_I_vs_I.dat')
        columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit", "$U_{SMU}$"]
        data_IvsI = main.read_data(path_IvsI, columns_IvsI)

        path_IlimitvsI = os.path.join(path, f'Channel_{channel}_Ilimit_vs_I.dat')
        columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
        data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

        x_0, y_0, l_0 = main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
        x_1, y_1, l_1 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
        x_2, y_2, l_2 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
        x_3, y_3, l_3 = main.get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
        x_4, y_4, l_4 = main.get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')

        x_err_0 = np.ones_like(x_0) * 3.05
        y_err_0 = main.SMU_V_error(y_0)
        x_err_1 = main.SMU_V_error(x_1)
        y_err_1 = np.ones_like(y_1) * 2.44
        x_err_2 = main.SMU_V_error(x_2)
        y_err_2 = np.ones_like(y_2) * 2.44
        x_err_3 = main.SMU_I_error(x_3, channel)
        y_err_3 = np.ones_like(y_3) * 2.44
        x_err_4 = np.ones_like(x_4) * 3.05
        y_err_4 = main.SMU_I_error(y_4, channel)

        constants = np.zeros(10)
        for n in range(10):
            constants[n] = float(config_root[f'{channel}'][f'{vars[n]}'])

        y_fit_0 = x_0 * constants[0]/10000 + constants[1]/100
        y_fit_1 = x_1 * constants[2] / 10000 + constants[3] / 100
        y_fit_2 = x_2 * constants[4] / 10000 + constants[5] / 100
        y_fit_3 = x_3 * constants[6] / 10000 + constants[7] / 100
        y_fit_4 = x_4 * constants[8] / 10000 + constants[9] / 100

        chi_square = np.zeros(5)
        chi_square[0] = np.sum((y_0-y_fit_0)**2/y_fit_0)
        chi_square[1] = np.sum((y_1 - y_fit_1) ** 2 / y_err_1)
        chi_square[2] = np.sum((y_2 - y_fit_2) ** 2 / y_err_2)
        chi_square[3] = np.sum((y_3 - y_fit_3) ** 2 / y_err_3)
        chi_square[4] = np.sum((y_4 - y_fit_4) ** 2 / y_err_4)

        red_chi = np.zeros_like(chi_square)
        red_chi[0] = chi_square[0] / (len(y_0)-2)
        red_chi[1] = chi_square[1] / (len(y_1) - 2)
        red_chi[2] = chi_square[2] / (len(y_2) - 2)
        red_chi[3] = chi_square[3] / (len(y_3) - 2)
        red_chi[4] = chi_square[4] / (len(y_4) - 2)

        print(chi_square)
        print(red_chi)

#clear_directory()
validate_outliers()
#compare_chi()