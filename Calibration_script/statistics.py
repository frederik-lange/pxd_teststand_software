'''
    In this program the outlier treatment was developed and tested
    Can be used for investigation of data with strange values
    The plots are deposited in the "statistics" subfolder in the file folder
'''
from Calibration_script import main
from Calibration_script import fit
import validation
import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import matplotlib.pyplot as plt
import os
import configparser

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
    plt.savefig(os.path.join(path,'statistics', title))
    return None

def plot_residuals(x,r, cut, m, n, title,name):
    plt.figure()
    plt.axhline(0)
    plt.grid()
    plt.xlabel("$I_{SMU}$")
    plt.ylabel("Residuals")
    plt.scatter(x[~cut], r[~cut], color='black')
    plt.scatter(x[cut], r[cut], color='grey')
    plt.title(title)
    plt.savefig(os.path.join(path,'statistics',name))
    return None

def outliers(x,y):
    # range criteria (remove saturation)
    ymax, ymin = np.amax(y), np.amin(y)
    range = ymax - ymin
    print(f"Maximum and minimum:", ymax, ymin)
    print("Range", range)
    upper_limit = ymin + 0.99 * range
    lower_limit = ymin + 0.01 * range
    print(f"Upper limit: {upper_limit}, lower limit: {lower_limit}")
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
    print(f"Mean of grad: {mean_grad}")
    print(f"Median of grad: {median_grad}")

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
    plt.savefig(os.path.join(path,f"statistics/Channel {channel}: 1st Gradient"))
    plt.figure()
    plt.grid()
    plt.xlabel("$I_{SMU}$ [mV]")
    plt.ylabel("$I_{OutMon}$ [mA]")
    plt.scatter(x[~help_cut], grad2[~help_cut], color="black")
    plt.scatter(x[help_cut], grad2[help_cut], color="grey")
    plt.title(f"Second Gradient of Channel {channel}")
    plt.savefig(os.path.join(path,f"statistics/Channel {channel}: 2nd Gradient"))

    # errors
    x_err = main.SMU_I_error(x,channel)
    y_err = np.ones_like(y)*2.44

    # auxiliary fit
    # Saturation got cut, considers all values along a line
    popt, pcov = so.curve_fit(lin, x[~help_cut], y[~help_cut], sigma=y_err[~help_cut], absolute_sigma=True)
    m, n = popt[0], popt[1]
    plot_with_fit(x[~help_cut],y[~help_cut],x[help_cut],y[help_cut],popt[0],popt[1],"$I_{SMU}$ [mA]","$I_{OutMon}$ [mV]",f"Channel {channel}: Plot with initial fit")
    print("Initial Fit:")
    print(f"a = {popt[0]} +/- {np.sqrt(pcov[0][0])}")
    print(f"b = {popt[1]} +/- {np.sqrt(pcov[1][1])}")

    # Fit with ODR
    popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~help_cut], y=y[~help_cut], x_err=x_err[~help_cut], y_err=y_err[~help_cut], p0=[popt[0],popt[1]])
    print(popt_odr)
    print(perr_odr)
    print("ODR Chi squared:",red_chi_2)
    print("ODR reduced chi squared", red_chi_2/(len(x[~cut])-2))
    m,n = popt_odr[0], popt_odr[1]

    # cut outliers
    # idea: create auxiliary fit and remove the worst values
    r = y - (m*x+n)
    std_r = np.std(r[~cut])
    mean_r = np.mean(r[~cut])
    print(f"Standard deviation of residuals: {std_r} \n Mean: {mean_r}")
    # cut on this
    # use abs of residuals because res are distribted around zero -> mean is useless
    #r = np.abs(r)
    cut1 = np.abs(r) > 2 * np.mean(np.abs(r))
    cut = cut + cut1

    print(f"Mean and median of abs(r): {np.mean(r)}, {np.median(r)}")
    plot_residuals(x,r, cut, m, n,f"Channel {channel}: Absolute values of residuals",f"Channel {channel}: Residuals")

    popt, perr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x[~cut], y=y[~cut], x_err=x_err[~cut], y_err=y_err[~cut], p0=[popt[0], popt[1]])
    print(popt)
    print(perr)
    print(red_chi_2)

    m, n = popt[0], popt[1]
    print("Final Fit:")
    print(f"a = {popt[0]} +/- {np.sqrt(pcov[0][0])}")
    print(f"b = {popt[1]} +/- {np.sqrt(pcov[1][1])}")
    plot_with_fit(x[~cut],y[~cut],x[cut],y[cut],popt[0],popt[1],"$I_{SMU}$ [mA]","$I_{OutMon}$ [mV]",f"Channel {channel}: Plot with final fit")
    print(f"ODR Chi Square: {red_chi_2}")

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

    return x[~cut], y[~cut],x[cut], y[cut]

channel = 13

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

# For Current
x,y,l = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
# For Limit
#x,y,l = main.get_and_prepare(data_IlimitvsI,"$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]")
# For Voltage
#x,y,l= main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')

plt.scatter(x,y)
plt.show()
plt.close()
x,y,x_cut,y_cut, m, n = outliers(x,y)

# compare to root:
root_gain, root_offset = -145443, 2478
y_fit = x * root_gain/10000 + root_offset/100
root_chisquare = np.sum((y - y_fit)**2/y_fit)
print(f"ROOT CHI SQUARE: {root_chisquare}")

# compare to old function:
x,y,l = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
x,y,x_cut,y_cut = outliers_old(x,y)
# errors
x_err = main.SMU_I_error(x,channel)
y_err = np.ones_like(y)*2.44
popt, pcov = so.curve_fit(lin, x, y, sigma=y_err, absolute_sigma=True)
popt, perr, red_chi_2 = fit.fit_odr(fit_func=lin, x=x, y=y, x_err=x_err, y_err=y_err, p0=[popt[0], popt[1]])
print(f"Parameters gained from the old function:\npopt: {popt}\nperr: {perr}\nChi square: {red_chi_2}")
