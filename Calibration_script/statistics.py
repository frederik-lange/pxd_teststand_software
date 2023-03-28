'''
    In this program the outlier treatment was developed and tested
    Can be used for investigation of data with strange values
    The plots are deposited in the "statistics" subfolder in the file folder
'''
from Calibration_script import main
from Calibration_script import fit
import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import matplotlib.pyplot as plt
import os
import configparser
import csv

path = '../data/example'
if not os.path.exists(os.path.join(path,'statistics')):
    os.mkdir(os.path.join(path,'statistics'))

lin = lambda x, a, b: a*x + b

def plot_with_fit(x,y,x_cut,y_cut,m,n,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, color='black')
    #plt.errorbar(x,y,yerr=2.44,fmt='ok')
    plt.scatter(x_cut, y_cut, color='grey')
    y_fit = lin(x,m,n)
    plt.plot(x,y_fit,'r-')
    y_fit = lin(x_cut,m,n)
    plt.plot(x_cut,y_fit,'r-')
    #plt.show()
    title = title.replace(" ","_")
    plt.savefig(os.path.join(path,'statistics', title))
    return None

def plot_residuals(x,r, cut, cutoff, title,name):
    plt.figure()
    plt.axhline(0)
    plt.grid()
    plt.xlabel("$I_{SMU}$")
    plt.ylabel("Residuals")
    plt.axhline(cutoff,color='r')
    plt.axhline(-cutoff,color='r')
    plt.scatter(x[~cut], r[~cut], color='black')
    plt.scatter(x[cut], r[cut], color='grey')
    plt.title(title)
    name = name.replace(" ","_")
    plt.savefig(os.path.join(path,'statistics',name))
    return None

def cut_outliers(x,y,channel):
    # range criteria (remove saturation)
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    #print(f"Maximum and minimum:", ymax, ymin)
    #print("Range", range)
    upper_limit = ymin + 0.99 * range
    lower_limit = ymin + 0.01 * range
    #print(f"Upper limit: {upper_limit}, lower limit: {lower_limit}")
    cut1 = y > upper_limit
    cut2 = y < lower_limit
    cut = cut1 + cut2

    # use gradient of y values to fit only linear values (remove scattering values)
    # gradient gives the "slope", twice gradient gives "curvature"
    grad = np.gradient(y)
    grad2 = np.gradient(grad)
    mean_grad = np.mean(grad[~cut])
    mean_grad2 = np.mean(grad2[~cut])
    median_grad = np.median(grad[~cut])
    median_grad2 = np.median(grad2[~cut])
    help_cut1 = grad2 > 10.0
    help_cut2 = grad2 < - 10.0
    help_cut = help_cut1 + help_cut2 + cut
    #print(f"Mean of grad: {mean_grad}")
    #print(f"Median of grad: {median_grad}")

    # remove false positive points of gradient criteria
    help_cut1 = np.abs(grad) > 1.2 * np.abs(mean_grad)
    help_cut2 = np.abs(grad) < 0.8 * np.abs(mean_grad)
    help_cut = help_cut + help_cut1 + help_cut2

    # if all points were removed due to outliers, use median instead of mean
    if x[~help_cut].size <= 1:
        print("Condition activated!")
        help_cut1 = grad2 > 10.0
        help_cut2 = grad2 < - 10.0
        help_cut = help_cut1 + help_cut2 + cut
        # remove false positive points of gradient criteria
        help_cut1 = np.abs(grad) > 1.2 * np.abs(median_grad)
        help_cut2 = np.abs(grad) < 0.8 * np.abs(median_grad)
        help_cut = help_cut + help_cut1 + help_cut2

    # plot gradients
    plt.figure()
    plt.grid()
    plt.xlabel("$I_{SMU}$ [mV]")
    plt.ylabel("$I_{OutMon}$ [mA]")
    plt.scatter(x[~help_cut], grad[~help_cut], color="black")
    plt.scatter(x[help_cut], grad[help_cut], color="grey")
    plt.title(f"First Gradient of Channel {channel}")
    plt.savefig(os.path.join(path,f"statistics/Channel_{channel}_1st_Gradient"))
    plt.figure()
    plt.grid()
    plt.xlabel("$I_{SMU}$ [mV]")
    plt.ylabel("$I_{OutMon}$ [mA]")
    plt.scatter(x[~help_cut], grad2[~help_cut], color="black")
    plt.scatter(x[help_cut], grad2[help_cut], color="grey")
    plt.title(f"Second Gradient of Channel {channel}")
    plt.savefig(os.path.join(path,f"statistics/Channel_{channel}_2nd_Gradient"))

    # errors
    x_err = main.SMU_I_error(x,channel)
    y_err = np.ones_like(y)*2.44

    # auxiliary fit
    # Saturation got cut, considers all values along a line
    popt, pcov = so.curve_fit(lin, x[~help_cut], y[~help_cut], sigma=y_err[~help_cut], absolute_sigma=True)
    m, n = popt[0], popt[1]
    plot_with_fit(x[~help_cut],y[~help_cut],x[help_cut],y[help_cut],popt[0],popt[1],"$I_{SMU}$ [mA]","$I_{OutMon}$ [mV]",f"Channel {channel} Plot with initial fit")
    #print("Initial Fit:")
    #print(f"a = {popt[0]} +/- {np.sqrt(pcov[0][0])}")
    #print(f"b = {popt[1]} +/- {np.sqrt(pcov[1][1])}")

    # Fit with ODR
    popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~help_cut], y=y[~help_cut], x_err=x_err[~help_cut], y_err=y_err[~help_cut], p0=[popt[0],popt[1]])
    #print(popt_odr)
    #print(perr_odr)
    #print("ODR Chi squared:",red_chi_2)
    #print("ODR reduced chi squared", red_chi_2/(len(x[~cut])-2))
    m,n = popt_odr[0], popt_odr[1]

    # cut outliers
    # idea: create auxiliary fit and remove the worst values
    r = y - (m*x+n)
    std_r = np.std(r[~cut])
    mean_r = np.mean(r[~cut])
    cutoff = 2 * np.mean(np.abs(r))
    #print(f"Standard deviation of residuals: {std_r} \n Mean: {mean_r}")
    # cut on this
    # use abs of residuals because res are distribted around zero -> mean is useless
    #r = np.abs(r)
    cut1 = np.abs(r) > cutoff
    cut = cut + cut1

    #print(f"Mean and median of abs(r): {np.mean(r)}, {np.median(r)}")
    plot_residuals(x,r, cut, cutoff,f"Channel {channel}: residuals",f"Channel {channel} Residuals")

    popt, perr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~cut], y=y[~cut], x_err=x_err[~cut], y_err=y_err[~cut], p0=[popt[0], popt[1]])
    #print(popt)
    #print(perr)
    #print(red_chi_2)

    m, n = popt[0], popt[1]
    #print("Final Fit:")
    #print(f"a = {popt[0]} +/- {np.sqrt(pcov[0][0])}")
    #print(f"b = {popt[1]} +/- {np.sqrt(pcov[1][1])}")
    plot_with_fit(x[~cut],y[~cut],x[cut],y[cut],popt[0],popt[1],"$I_{SMU}$ [mA]","$I_{OutMon}$ [mV]",f"Channel {channel} Plot with final fit")
    #print(f"ODR Chi Square: {red_chi_2}")

    return x[~cut], y[~cut], x[cut], y[cut], m, n

def outliers_old(x, y):
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

    # Making array same size as data with only False in it
    cut = np.zeros_like(slopes, dtype=bool)
    # Set False to zero in parts where data gradient is close to zero
    cut[:][np.isclose(np.gradient(y), 0, atol=tolerance)] = True

    if x[~cut].size == 0:
        return x[~cut], y[~cut], x[cut], y[cut]
    else:
        # calculate mean and standard deviation
        mean, std = np.mean(slopes[~cut]), np.std(slopes[~cut])
        #print(f'For slopes: mean: {mean}, standard deviation: {std}')
        #cut to  2 sigma
        cut[np.logical_or((slopes >= 2 * std + mean), (slopes <= mean - 2 * std))] = True

    return x[~cut], y[~cut],x[cut], y[cut], cut

def outliers_new(x, y, x_err, y_err):
    # range criteria (remove saturation)
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    upper_limit = ymin + 0.99 * range
    lower_limit = ymin + 0.01 * range
    cut1 = y > upper_limit
    cut2 = y < lower_limit
    cut = cut1 + cut2

    # use gradient of y values to fit only linear values (remove scattering values)
    # gradient gives the "slope", twice gradient gives "curvature"
    grad = np.gradient(y)
    grad2 = np.gradient(grad)
    mean_grad = np.mean(grad[~cut])
    mean_grad2 = np.mean(grad2[~cut])
    median_grad = np.median(grad[~cut])
    median_grad2 = np.median(grad2[~cut])
    help_cut1 = grad2 > 10.0
    help_cut2 = grad2 < - 10.0
    help_cut = help_cut1 + help_cut2 + cut

    # remove false positive points of gradient criteria
    help_cut1 = np.abs(grad) > 1.2 * np.abs(mean_grad)
    help_cut2 = np.abs(grad) < 0.8 * np.abs(mean_grad)
    help_cut = help_cut + help_cut1 + help_cut2

    # if all points were removed due to outliers, use median instead of mean
    if x[~help_cut].size <= 1:
        help_cut1 = grad2 > 10.0
        help_cut2 = grad2 < - 10.0
        help_cut = help_cut1 + help_cut2 + cut
        # remove false positive points of gradient criteria
        help_cut1 = np.abs(grad) > 1.2 * np.abs(median_grad)
        help_cut2 = np.abs(grad) < 0.8 * np.abs(median_grad)
        help_cut = help_cut + help_cut1 + help_cut2

    if x[~help_cut].size >= 2:
        # auxiliary fit
        # Saturation got cut, considers all values along a line
        popt, pcov = so.curve_fit(lin, x[~help_cut], y[~help_cut], sigma=y_err[~help_cut], absolute_sigma=True)
        m, n = popt[0], popt[1]

        # Fit with ODR
        popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~help_cut], y=y[~help_cut],
                                                    x_err=x_err[~help_cut], y_err=y_err[~help_cut],
                                                    p0=[popt[0], popt[1]])
        m, n = popt_odr[0], popt_odr[1]

        # cut outliers
        r = y - (m * x + n)
        std_r = np.std(r[~cut])
        mean_r = np.mean(r[~cut])
        # use abs of residuals because res are distributed around zero -> mean is useless
        r = np.abs(r)
        # cut on this
        cut1 = np.abs(r) > 2 * np.abs(np.mean(r))
        cut = cut + cut1

        # Final Fit
        popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~cut], y=y[~cut], x_err=x_err[~cut],
                                                    y_err=y_err[~cut], p0=[popt[0], popt[1]])
        m, n = popt_odr[0], popt_odr[1]
        #print(f"Chi square: {red_chi_2}")
        return x[~cut], y[~cut], x[cut], y[cut], cut
    else:
        print('Too many values cut!')
        return x[~help_cut], y[~help_cut], x, y, np.ones_like(x, dtype=bool)

def outliers():
    for channel in [5,21]:
        path_UvsU = os.path.join(path,f'Channel_{channel}_U_vs_U.dat')
        # Test File
        #path_UvsU = f"./../data/statistics/Channel_{channel}_U_vs_U_test.dat"
        columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
        data_UvsU = main.read_data(path_UvsU, columns_UvsU)

        path_IvsI = os.path.join(path,f"Channel_{channel}_I_vs_I.dat")
        # Test File
        #path_IvsI = f"./../data/statistics/Channel_{channel}_I_vs_I_test.dat"
        columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
        data_IvsI = main.read_data(path_IvsI, columns_IvsI)

        path_IlimitvsI = os.path.join(path,f'Channel_{channel}_Ilimit_vs_I.dat')
        columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
        data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

        x_0, y_0, l_0 = main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
        x_1, y_1, l_1 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
        x_2, y_2, l_2 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
        x_3, y_3, l_3 = main.get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
        x_4, y_4, l_4 = main.get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')

        if channel == 5:
            x_3,y_3,x_cut_3,y_cut_3, m, n = cut_outliers(x_3,y_3,channel)
        if channel == 21:
            x_0, y_0, x_cut_0, y_cut_0, m, n = cut_outliers(x_0, y_0,channel)

def get_chisquare(x,y,dx,dy):
    # p0 estimator
    popt, pcov = so.curve_fit(lin, x, y, sigma=dy, absolute_sigma=True)

    # (m, b), (SSE,), *_ = np.polyfit(x, y, deg=1, full=True)
    popt, perr, red_chi_2 = fit.fit_odr(fit_func=lin,x=x,y=y,x_err=dx,y_err=dy,p0=[popt[0], popt[1]])
    return red_chi_2

def compare_chi():
    # compare chi square values of the different methods
    chi_squares_new = np.zeros(120)
    chi_squares_new = np.reshape(chi_squares_new,(24,5))
    chi_squares_old = np.zeros(120)
    chi_squares_old = np.reshape(chi_squares_old, (24, 5))
    config_root = configparser.ConfigParser()
    config_root.read('../data/example/constants_from_root.ini')
    path = '../data/example'
    vars = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET', 'ADC_U_REGULATOR_GAIN',
            'ADC_U_REGULATOR_OFFSET',
            'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    for channel in range(24):
        print(f'Channel {channel}')
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


        # for new method
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

        x_0,y_0,x_cut_0,y_cut_0,cut_0 = outliers_new(x_0,y_0,x_err_0,y_err_0)
        x_1, y_1, x_cut_1, y_cut_1, cut_1 = outliers_new(x_1, y_1, x_err_1, y_err_1)
        x_2, y_2, x_cut_2, y_cut_2, cut_2 = outliers_new(x_2, y_2, x_err_2, y_err_2)
        x_3, y_3, x_cut_3, y_cut_3, cut_3 = outliers_new(x_3, y_3, x_err_3, y_err_3)
        x_4, y_4, x_cut_4, y_cut_4, cut_4 = outliers_new(x_4, y_4, x_err_4, y_err_4)

        chi_squares_new[channel,0] = get_chisquare(x_0,y_0,x_err_0[~cut_0],y_err_0[~cut_0])
        chi_squares_new[channel,1] = get_chisquare(x_1, y_1, x_err_1[~cut_1], y_err_1[~cut_1])
        chi_squares_new[channel,2] = get_chisquare(x_2, y_2, x_err_2[~cut_2], y_err_2[~cut_2])
        chi_squares_new[channel,3] = get_chisquare(x_3, y_3, x_err_3[~cut_3], y_err_3[~cut_3])
        chi_squares_new[channel,4] = get_chisquare(x_4, y_4, x_err_4[~cut_4], y_err_4[~cut_4])


        # for old method
        x_0, y_0, l_0 = main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
        x_1, y_1, l_1 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
        x_2, y_2, l_2 = main.get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
        x_3, y_3, l_3 = main.get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
        x_4, y_4, l_4 = main.get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')

        x_0, y_0, x_cut_0, y_cut_0, cut_0 = outliers_old(x_0, y_0)
        x_1, y_1, x_cut_1, y_cut_1, cut_1 = outliers_old(x_1, y_1)
        x_2, y_2, x_cut_2, y_cut_2, cut_2 = outliers_old(x_2, y_2)
        x_3, y_3, x_cut_3, y_cut_3, cut_3 = outliers_old(x_3, y_3)
        x_4, y_4, x_cut_4, y_cut_4, cut_4 = outliers_old(x_4, y_4)

        chi_squares_old[channel, 0] = get_chisquare(x_0, y_0, x_err_0[~cut_0], y_err_0[~cut_0])
        chi_squares_old[channel, 1] = get_chisquare(x_1, y_1, x_err_1[~cut_1], y_err_1[~cut_1])
        chi_squares_old[channel, 2] = get_chisquare(x_2, y_2, x_err_2[~cut_2], y_err_2[~cut_2])
        chi_squares_old[channel, 3] = get_chisquare(x_3, y_3, x_err_3[~cut_3], y_err_3[~cut_3])
        chi_squares_old[channel, 4] = get_chisquare(x_4, y_4, x_err_4[~cut_4], y_err_4[~cut_4])

    print(chi_squares_old)
    print(chi_squares_new)
    with open('../data/Chi_square_example.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel','old','new','old','new','old','new','old','new','old','new'])
        for n in range(24):
            list = [n]
            for v in range(5):
                list.append((chi_squares_old[n,v]))
                list.append((chi_squares_new[n,v]))
            print(list)
            writer.writerow(list)

    '''
        constants = np.zeros(10)
        for n in range(10):
            constants[n] = float(config_root[f'{channel}'][f'{vars[n]}'])

        y_fit_0 = x_0 * constants[0]/10000 + constants[1]/100
        y_fit_1 = x_1 * constants[2] / 10000 + constants[3] / 100
        y_fit_2 = x_2 * constants[4] / 10000 + constants[5] / 100
        y_fit_3 = x_3 * constants[6] / 10000 + constants[7] / 100
        y_fit_4 = x_4 * constants[8] / 10000 + constants[9] / 100

        chi_square = np.zeros(5)
        chi_square[0] = np.sum((y_0-y_fit_0)**2/y_fit_0**2)
        chi_square[1] = np.sum((y_1 - y_fit_1) ** 2 / y_err_1**2)
        chi_square[2] = np.sum((y_2 - y_fit_2) ** 2 / y_err_2**2)
        chi_square[3] = np.sum((y_3 - y_fit_3) ** 2 / y_err_3**2)
        chi_square[4] = np.sum((y_4 - y_fit_4) ** 2 / y_err_4**2)

        red_chi = np.zeros_like(chi_square)
        red_chi[0] = chi_square[0] / (len(y_0)-2)
        red_chi[1] = chi_square[1] / (len(y_1) - 2)
        red_chi[2] = chi_square[2] / (len(y_2) - 2)
        red_chi[3] = chi_square[3] / (len(y_3) - 2)
        red_chi[4] = chi_square[4] / (len(y_4) - 2)

        #print(chi_square)
        print(red_chi)

        x_3, y_3, x_cut_3, y_cut_3, m, n = cut_outliers(x_3, y_3)
        x_err_3 = main.SMU_I_error(x_3, channel)
        y_err_3 = np.ones_like(y_3) * 2.44
        popt, pcov = so.curve_fit(lin, x_3, y_3, sigma=y_err_3, absolute_sigma=True)
        popt, perr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x_3, y=y_3, x_err=x_err_3, y_err=y_err_3, p0=[popt[0], popt[1]])
        print(red_chi_2)
        y_fit_3 = popt[0]*x_3 + popt[1]
        #print(y_3 - y_fit_3)
        chi_sq_3 = np.sum((y_3 - y_fit_3) ** 2 / y_err_3 ** 2) / (len(y_3)-2)
        print(chi_sq_3)
        '''

#outliers()
compare_chi()