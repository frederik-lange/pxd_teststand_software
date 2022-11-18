from Calibration_script import main
import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import matplotlib.pyplot as plt
import os
import configparser

lin = lambda x, a, b: a*x + b

def plot_with_fit(x,y,x_cut,y_cut,m,n,xlabel,ylabel,title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, color='black')
    plt.scatter(x_cut, y_cut, color='grey')
    y_fit = lin(x,m,n)
    plt.plot(x,y_fit,'r-')
    #plt.show()
    plt.savefig(os.path.join('../data/validation', title))
    return None

def plot_residuals(x,r, cut, m, n, title):
    plt.figure()
    plt.axhline(0)
    plt.grid()
    plt.scatter(x[~cut], r[~cut], color='black')
    plt.scatter(x[cut], r[cut], color='grey')
    plt.title(title)
    plt.savefig(os.path.join('../data/validation',title))
    return None

def chi_square(y,y_fit):
    chi2 = np.sum( (y-y_fit)**2/y_fit )
    return chi2

def outliers(x,y):
    # range criteria
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
    grad = np.gradient(y)
    grad2 = np.gradient(grad)
    mean_grad = np.mean(grad)
    mean_grad2 = np.mean(grad2)
    median_grad = np.median(grad)
    median_grad2 = np.median(grad2)
    help_cut1 = grad2 > 10.0
    help_cut2 = grad2 < - 10.0
    help_cut = help_cut1 + help_cut2

    print(mean_grad)
    print(median_grad)

    # remove false positive points of gradient criteria
    help_cut1 = grad > 1.2 * mean_grad
    help_cut2 = grad < 0.8 * mean_grad2
    help_cut = help_cut + help_cut1 + help_cut2

    # plot gradients
    plt.figure()
    plt.grid()
    plt.scatter(x[~help_cut], grad2[~help_cut], color="black")
    plt.scatter(x[help_cut], grad2[help_cut], color="grey")
    plt.title("Twice Gradient")

    plt.figure()
    plt.grid()
    plt.scatter(x[~help_cut], grad[~help_cut], color="black")
    plt.scatter(x[help_cut], grad[help_cut], color="grey")
    plt.title("Gradient")
    plt.show()

    # auxiliary fit
    popt, pcov = so.curve_fit(lin, x[~help_cut], y[~help_cut])
    m, n = popt[0], popt[1]
    plot_with_fit(x[~help_cut],y[~help_cut],x[help_cut],y[help_cut],popt[0],popt[1],"","",f"Channel {channel}: Plot with auxiliary fit")

    #model = ss.linregress(x[~cut],y[~cut])
    #print(f"R^2 value: {model.rvalue ** 2}\nSlope: {model.slope}, Intercept:  {model.intercept}")

    # cut outliers
    # idea: create auxiliary fit and remove 5% of worst values
    # then repeat
    iterations = [1,2]
    for i in iterations:
        r = y - (m*x+n)
        std_r = np.std(r[~cut])
        mean_r = np.mean(r[~cut])
        #print(f"Standard deviation of residuals: {std_r} \n Mean: {mean_r}")
        y_fit = m*x+n
        chi2 = chi_square(y,y_fit)
        chi2_red = chi2/(len(y)-2)
        print(f"Chi square: {chi2}\n Reduced Chi square: {chi2_red}")
        # cut on this
        """ First method:
        cut1 = r > 2*std_r
        cut2 = r<(-2)*std_r
        cut = cut + cut1 + cut2
        print(cut)
        """
        """ Second method
        NOTES: Aux linear fit works really well
        cutting outliers via residuals not
        """
        r = np.abs(r)
        cut1 = np.abs(r) > 2 * np.abs(np.mean(r))
        cut = cut + cut1

        print(np.mean(r),np.median(r))
        plot_residuals(x,r, cut, m, n,f"Channel {channel}: Residuals after iteration {i}")
        popt, pcov = so.curve_fit(lin, x[~cut], y[~cut])
        m, n = popt[0], popt[1]
        print(f"a = {popt[0]} +/- {np.sqrt(pcov[0][0])}")
        print(f"b = {popt[1]} +/- {np.sqrt(pcov[1][1])}")
        plot_with_fit(x[~cut],y[~cut],x[cut],y[cut],popt[0],popt[1],"","",f"Channel {channel}: Plot with fit after iteration {i}")
        model = ss.linregress(x[~cut], y[~cut])
        #print(f"R^2 value: {model.rvalue ** 2}\nSlope: {model.slope}, Intercept:  {model.intercept}")
        y_fit = m * x + n
        chi2 = chi_square(y[~cut], y_fit[~cut])
        chi2_red = chi2 / (len(y[~cut]) - 2)
        print(f"Chi square: {chi2}\n Reduced Chi square: {chi2_red}")

    return x[~cut], y[~cut], x[cut], y[cut], m, n

### important
channel = 5
###

path_UvsU = f"./../data/Channel_{channel}_U_vs_U.dat"
columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
data_UvsU = main.read_data(path_UvsU, columns_UvsU)

path_IvsI = f"./../data/Channel_{channel}_I_vs_I.dat"
#path_IvsI = f"./../data/validation/Channel_5_test_I.dat"
columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
data_IvsI = main.read_data(path_IvsI, columns_IvsI)

path_IlimitvsI = f"./../data/Channel_{channel}_Ilimit_vs_I.dat"
columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
data_IlimitvsI = main.read_data(path_IlimitvsI, columns_IlimitvsI)

x,y,l = main.get_and_prepare(data_IvsI,'$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
#x,y,l= main.get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
x,y,x_cut,y_cut, m, n = outliers(x,y)
#validation.scatter_cut(x,y,x_cut,y_cut,"x","y",f"Channel {channel}: new")