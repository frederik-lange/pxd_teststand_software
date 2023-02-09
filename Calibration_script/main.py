import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import scipy.optimize as so
import fit
from matplotlib.backends.backend_pdf import PdfPages
import csv
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import datetime
import database

config = configparser.ConfigParser()
config_ini = configparser.ConfigParser()
config_err = configparser.ConfigParser()
config_ini.optionxform = str
config_err.optionxform = str

def prepare_data(x, y):
    """
    Excluding none and inf in data
    :param x: calibration data
    :param y: calibration data
    :return: calibration data without nan and inf
    """
    cut = np.zeros_like(x, dtype=bool)
    cut[np.isinf(x) | np.isnan(x)] = True
    cut[np.isinf(y) | np.isnan(y)] = True
    cut[x == 0] = True
    return x[~cut], y[~cut]

def cut_outliers(x, y, x_err, y_err, channel):
    """
    To work properly this function requires values that are normally distributed around a linear slope
    The function determines a straight line among the values and fits to them.
    The residuals are computed for all points and the points too far away are removed.
    Then the final fit is performed.
    """

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
        popt, pcov = so.curve_fit(linear, x[~help_cut], y[~help_cut], sigma=y_err[~help_cut], absolute_sigma=True)
        m, n = popt[0], popt[1]

        # Fit with ODR
        popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=linear, x=x[~help_cut], y=y[~help_cut], x_err=x_err[~help_cut], y_err=y_err[~help_cut], p0=[popt[0], popt[1]])
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
        popt_odr, perr_odr, red_chi_2 = fit.fit_odr(fit_func=linear, x=x[~cut], y=y[~cut], x_err=x_err[~cut], y_err=y_err[~cut], p0=[popt[0], popt[1]])
        m, n = popt_odr[0], popt_odr[1]
        return x[~cut], y[~cut], x[cut], y[cut], cut
    else:
        print('Too many values cut!')
        return x[~help_cut],y[~help_cut], x,y,np.ones_like(x, dtype=bool)

def linear(m,x,b):
    """
    Defining the form of a linear function
    :param m: slope
    :param x: calibration data
    :param b: offset
    :return: returns form of linear function
    """
    return m*x+b

def get_and_prepare(data,x_data,y_data):
    x_1 = data[x_data]
    y_1 = data[y_data]
    length = len(x_1)
    x_2,y_2 = prepare_data(x_1, y_1)

    return x_2,y_2, length

def plot_and_fit(x, y, dx, dy, x_cut, y_cut, xlabel, ylabel, label,n):
    """
    Creates plots from cleaned calibration data and linear regression
    :param x: Calibration Data
    :param y: Calibration Data
    :param dx: error
    :param dy: error
    :param xlabel: Label of x axis
    :param ylabel:  label of y axis
    :param label: title of plot
    :return: plot, slope and offset
    """
    if x.size <= 1:
        plt.subplot(2, 3, n)
        plt.scatter(x_cut, y_cut, color='grey', marker='.', linewidths=1.0, label='Outliers')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(prop={'size': 8})
        # default values
        m = 10000
        b = 0
        m_err = 0
        b_err = 0
        return m, b, m_err, b_err
    else:
        # p0 estimator
        p0 = np.mean((y-y[0])/x)

        # (m, b), (SSE,), *_ = np.polyfit(x, y, deg=1, full=True)
        popt, perr, red_chi_2 = fit.fit_odr(fit_func=linear,
                                            x=x,
                                            y=y,
                                            x_err=dx,
                                            y_err=dy,
                                            p0=[p0,y[0]])

        # Make result string
        res = "Fit results of {}:\n".format(fit.fit_odr.__name__)
        res += "m = ({:.3E} {} {:.3E})".format(popt[0], u'\u00B1', perr[0]) + " \n"
        res += "b = ({:.3E} {} {:.3E})".format(popt[1], u'\u00B1', perr[1]) + " \n"
        res += "red. Chi^2 = {:.3f}".format(red_chi_2) + "\n"
        m = popt[0]
        b = popt[1]
        m_err = perr[0]
        b_err = perr[1]

        # plot each of 5 subplots
        plt.subplot(2, 3, n)
        plt.scatter(x, y, color='k', marker='.', linewidths=1.0 )
        plt.scatter(x_cut, y_cut, color='grey',marker='.', linewidths=1.0, label = 'Outliers')
        plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='none', ecolor='k', label = label)
        plt.plot(x, m * x + b, color='r', label=f'slope = {round(m, 4)}''\n' f'int= {round(b, 4)}''\n'r' $\chi^2_{\mathrm{red}}$ =' f'{round(red_chi_2, 4)}')
        plt.rcParams["figure.autolayout"] = True
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(prop={'size':8})
        return m, b, m_err, b_err

def residual_plots(data_x, data_y,x,y,cut_x, cut_y, m, b, n, channel):
    """
    Plots the rediuals for the data points
    :param data_x: Name of the x data
    :param data_y: Name of the y data
    :param x: the x data without the outliers
    :param y: the y data without the outliers
    :param cut_x: the x outliers
    :param cut_y: the y outliers
    :param m: the slope
    :param b: the offset
    :param n: which subplot
    :return: residual plots
    """
    plt.subplot(2,3,n)
    r = y-((m * x) + b)
    plt.scatter(x, r , color='g', marker='.', linewidths=1.0, label="Residual:\n"+data_x+" vs. "+data_y)
    plt.scatter(cut_x, cut_y-(m * cut_x + b) , color='grey', marker='.', linewidths=1.0)
    plt.xlabel(data_x)
    plt.ylabel(data_y)
    plt.legend(prop={'size': 8})

def help_plots(data, data_x, data_y, title,n):
    x_0 = data[data_x]
    y_0 = data[data_y]
    plt.subplot(2, 3, n)
    plt.scatter(x_0, y_0, color='b', marker='.', linewidths=1.0, label=title)
    plt.xlabel(data_x)
    plt.ylabel(data_y)
    plt.legend(prop={'size': 8})

def read_data(path, columns):
    """
    Gets data from path
    :param path: path to .dat diles
    :param columns: multiple columns in each file
    :return: data
    """
    dtype = [(col, '<f8') for col in columns]
    data = np.loadtxt(fname=path, delimiter=" ", skiprows=0, dtype=dtype)
    return data

def SMU_V_error(data):
    error=np.zeros_like(data)

    for i in range(len(error)):
        if (abs(data[i])<200):
            error[i] =  data[i]*0.00015+0.225
        elif (abs(data[i])<2000):
            error[i] = data[i]*0.00015+0.325
        elif (abs(data[i])< 20000):
            error[i] = data[i]*0.00015+5
        elif (abs(data[i])< 200000):
            error[i] = data[i]*0.00015+50
    return error

def SMU_I_error(data, channel):
    error=np.zeros_like(data)
    if channel == 13:
        np.multiply(data, 0.001) # previously: 0.0001
    for i in range(len(error)):
        if (abs(data[i])<0.1):
            error[i] =  data[i]*0.0002+0.000025
        elif (abs(data[i])<1):
            error[i] = data[i]*0.0002+0.0002
        elif (abs(data[i])< 10):
            error[i] = data[i]*0.0002+0.0025
        elif (abs(data[i])< 100):
            error[i] = data[i]*0.0002+0.02
        elif (abs(data[i]) < 1000):
            error[i] = data[i] * 0.0003 + 1.5
        elif (abs(data[i]) < 1500):
            error[i] = data[i] * 0.0005 + 3.5
        elif (abs(data[i]) < 3000):
            error[i] = data[i] * 0.04 + 7
        elif (abs(data[i]) < 10000):
            error[i] = data[i] * 0.04 + 25
    return error

def histo_deleted_points(length,deleted_points):
    plt.subplots(figsize=(12, 6))
    channels = np.arange(0,24,1)
    plot_histo(channels, deleted_points[:,0], '$U_{out} vs. U_{DAC}$',1, length )
    plot_histo(channels, deleted_points[:,1], '$U_{regulator} vs. U_{out}$', 2, length)
    plot_histo(channels, deleted_points[:,2], '$U_{load} vs. U_{out}$', 3, length)
    plot_histo(channels, deleted_points[:,3],'$I_{outMON} vs. I_{SMU}$', 4, length)
    plot_histo(channels, deleted_points[:,4],'$I_{lim,SMU} vs. I_{lim,DAC}$', 5, length)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.6)

    plt.savefig(os.path.join(config["calibration_data"].get("data_path"),'deleted_point.pdf'))
    plt.close()

def plot_histo(x,y,title,n, length):
    color_map = cm.get_cmap('RdYlGn_r')

    data_hight_normalized = [z / length for z in y]
    colors = color_map(data_hight_normalized)

    plt.subplot(2, 3, n)
    plt.bar(x, y, color=colors, width=0.72, label=title)
    plt.xlabel('Channel')
    plt.xticks(fontsize=6)
    plt.ylabel('Deleted Points')
    plt.legend(prop={'size': 8})
    plt.rcParams["figure.autolayout"] = True

    sm = ScalarMappable(cmap=color_map, norm=plt.Normalize(0, length))
    sm.set_array([])
    plt.colorbar(sm)
    #cbar.set_label('Color', rotation=270, labelpad=25)

def pass_fail(l_1, deleted_points):
    too_many_deleted = False
    for channel in range(24):
        l = l_1*0.6
        if deleted_points[channel][0] > l or deleted_points[channel][1] > l or deleted_points[channel][2] > l or deleted_points[channel][3] > l or deleted_points[channel][4] > l:
            print(f'Warning! Please check Channel {channel}. Too many points were deleted.')
            too_many_deleted = True
    # Compare channels, improve output (make it more clearly)
    # if successful: add to database
    wrong_channels = np.zeros(24)
    for channel in range(24):
        add = True
        for name in ['DAC_VOLTAGE_GAIN','DAC_VOLTAGE_OFFSET','ADC_U_LOAD_GAIN','ADC_U_LOAD_OFFSET','ADC_U_REGULATOR_GAIN','ADC_U_REGULATOR_OFFSET',
                     'ADC_I_MON_GAIN','ADC_I_MON_OFFSET','DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET']:
            if check_range(name,channel) == False:
                wrong_channels[channel] = 1

    #print('\n'.join(map(str, residuals))) #what does this do?
    success = False
    if too_many_deleted == True:
        print('\nCalibration was NOT successful! Too many points were deleted!')
        success = False
    elif np.sum(wrong_channels) > 0:
        print('\nCalibration was NOT successful! Please check warnings!')
        success = False
    else:
        print('\nCalibration was successful!')
        success = True

    # write calibration result in ini file
    with open(os.path.join(config["calibration_data"].get("data_path"), 'constants.ini'), 'w') as configfile:
        if success == True:
            config_ini["Information"]["success"] = "True"
        else:
            config_ini["Information"]["success"] = "False"
        config_ini.write(configfile)
    return success

def check_range(name,channel):
    in_range = False
    value = float(config_ini[f'{channel}'].get(name))
    config_vals = configparser.ConfigParser()
    config_errs = configparser.ConfigParser()
    config_vals.read('../data/database.ini')
    config_errs.read('../data/database_std.ini')
    mean = float(config_vals[f'{channel}'][f'{name}_{channel}'])
    std = float(config_errs[f'{channel}'][f'{name}_{channel}'])
    upper, lower = mean + 4*std, mean - 4*std
    if value > upper or value < lower:
        in_range = False
        print(f'Warning! Please check channel {channel}. {name} out of usual range!')
    else:
        in_range = True

    return in_range

def write_in_ini(ini,channel,m0,b0,m1,b1,m2,b2,m3,b3,m4,b4):
    ini[f'{channel}'] =  {'DAC_VOLTAGE_GAIN': round(m0 * 10000, 0),
                                'DAC_VOLTAGE_OFFSET': round(b0 * 100, 0),
                                'ADC_U_LOAD_GAIN': round(m1 * 10000, 0),
                                'ADC_U_LOAD_OFFSET': round(b1 * 100, 0),
                                'ADC_U_REGULATOR_GAIN': round(m2 * 10000, 0),
                                'ADC_U_REGULATOR_OFFSET': round(b2 * 100, 0),
                                'ADC_I_MON_GAIN': round(m3 * 10000, 0),
                                'ADC_I_MON_OFFSET': round(b3 * 100, 0),
                                'DAC_CURRENT_GAIN': round(m4 * 10000, 0),
                                'DAC_CURRENT_OFFSET': round(b4 * 100, 0)}

def add_to_database(path):
    s = input('Do you want to add the new constants to the database? (yes/no)')
    if s == 'yes' or s == 'y':
        ps = input('Please enter the number of the power supply:')
        with open('../data/database.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=database.names)
            database.add_constants(writer, os.path.join(path, 'constants.ini'), ps)
            print("The calibration constants were added to the database.")
        database.update_range()

def main():
    # Getting path from .ini file
    config.read("path.ini")
    # Making the folders for the  plots
    if os.path.exists(os.path.join(config["calibration_data"].get("data_path"),"plots")) == True:
        pass
    else:
        os.mkdir(os.path.join(config["calibration_data"].get("data_path"),"plots"))

    deleted_points = np.zeros([24,5])
    for channel in range(24):
        path = os.path.join(config["calibration_data"].get("data_path"),f"plots/Channel_{channel}.pdf")

        with PdfPages(path) as pdf:
            """
            The graphs:
            0) U Cal: Uset vs. Uout (x_axis: U_dac[mV], y_axis: U_output[mV]) 
            1) U Cal: Uout vs. MonUreg (x_axis: U_output[mV], y_axis: U_regulator[mV])
            2) U Cal: Uout vs. MonUload (x_axis: U_output[mV], y_axis: U_load[mV])
            3) I Cal: Iout vs. IoutMon  (x_axis: I_output[mA], y_axis: I_monitoring[mA])
            4) I Cal: DAC limit vs. IMeasured  (x_axis: Limit DAC[mV], y_axis: limit current[mA])
            """
            plt.subplots(figsize=(12, 6))

            # Opening the U_vs_U ini file
            path_UvsU = os.path.join(config["calibration_data"].get("data_path"), 'Channel_%d_U_vs_U' % channel + '.dat')
            columns_UvsU = ["$U_{DAC}$ [mV]", "$U_{out}$ [mV]", "$U_{regulator}$ [mV]", "$U_{load}$ [mV]", "unknown 5","unknown 6"]
            data_UvsU = read_data(path_UvsU, columns_UvsU)

            # get file modification time on linux
            timestamp = os.path.getmtime(path_UvsU)
            m_time = datetime.date.fromtimestamp(timestamp)

            # Opening I_vs_I.dat file
            path_IvsI = os.path.join(config["calibration_data"].get("data_path"),'Channel_%d_I_vs_I' % channel + '.dat')
            columns_IvsI = ["unknown 1", "$I_{out(SMU)}$ [mA]", "$I_{outMon}$ [mV]", "$U_{outMon}$", "StatBit","$U_{SMU}$"]
            data_IvsI = read_data(path_IvsI, columns_IvsI)

            # Opening Ilimit_vs_I.dat file
            path_IlimitvsI = os.path.join(config["calibration_data"].get("data_path"),'Channel_%d_Ilimit_vs_I' % channel + '.dat')
            columns_IlimitvsI = ["$I_{lim,DAC}$ [mV]", "$I_{lim,SMU}$ [mA]", "unknown 3", "unknown 4", "StatBit"]
            data_IlimitvsI = read_data(path_IlimitvsI, columns_IlimitvsI)


            # 0) Plot(U Cal: Uset vs. U out)
            x_0,y_0,l_0= get_and_prepare(data_UvsU, '$U_{DAC}$ [mV]', '$U_{out}$ [mV]')
            x_err_0 = np.ones_like(x_0)*3.05
            y_err_0 = SMU_V_error(y_0)
            x_0, y_0,x_cut_0,y_cut_0, cut_0 = cut_outliers(x_0, y_0, x_err_0, y_err_0, channel)
            m_0, b_0, m_err_0, b_err_0 = plot_and_fit(x_0, y_0, x_err_0[~cut_0], y_err_0[~cut_0], x_cut_0,y_cut_0,  '$U_{DAC}$ [mV]', '$U_{out}$ [mV]', '$U_{out} vs. U_{DAC}$',1)

            # 1) U Cal: Uout vs. MonUreg
            x_1,y_1, l_1= get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]')
            x_err_1 = SMU_V_error(x_1)
            y_err_1 = np.ones_like(y_1)*2.44
            x_1, y_1,x_cut_1,y_cut_1, cut_1 = cut_outliers(x_1, y_1, x_err_1, y_err_1, channel)
            m_1, b_1, m_err_1, b_err_1 = plot_and_fit(x_1, y_1, x_err_1[~cut_1], y_err_1[~cut_1],x_cut_1,y_cut_1, '$U_{out}$ [mV]', '$U_{regulator}$ [mV]', '$U_{regulator} vs. U_{out}$',2)

            # 2) U Cal: Uout vs. MonUload
            x_2,y_2, l_2= get_and_prepare(data_UvsU, '$U_{out}$ [mV]', '$U_{load}$ [mV]')
            x_err_2 = SMU_V_error(x_2)
            y_err_2 = np.ones_like(y_2)*2.44
            x_2, y_2, x_cut_2,y_cut_2, cut_2 = cut_outliers(x_2, y_2, x_err_2, y_err_2, channel)
            m_2, b_2, m_err_2, b_err_2 = plot_and_fit(x_2, y_2, x_err_2[~cut_2],y_err_2[~cut_2], x_cut_2,y_cut_2, '$U_{out}$ [mV]', '$U_{load}$ [mV]', '$U_{load} vs. U_{out}$',3)

            # 3) I Cal: Iout vs. IoutMon
            x_3,y_3, l_3 = get_and_prepare(data_IvsI, '$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]')
            x_err_3 = SMU_I_error(x_3, channel)
            y_err_3 = np.ones_like(y_3) * 2.44
            # special monitoring for channel 13 (uA)
            if channel == 13:
                x_3 = x_3 * 1000
                x_err_3 = x_err_3 * 1000
            x_3, y_3, x_cut_3,y_cut_3, cut_3 = cut_outliers(x_3, y_3, x_err_3, y_err_3, channel)
            if channel == 13:
                m_3, b_3, m_err_3, b_err_3 = plot_and_fit(x_3, y_3, x_err_3[~cut_3], y_err_3[~cut_3], x_cut_3,y_cut_3, '$I_{SMU}$ [$\mu$A]', '$I_{outMon}$ [mV]', '$I_{outMON} vs. I_{SMU}$',4)
            else:
                m_3, b_3, m_err_3, b_err_3 = plot_and_fit(x_3, y_3, x_err_3[~cut_3], y_err_3[~cut_3], x_cut_3,y_cut_3, '$I_{SMU}$ [mA]', '$I_{outMon}$ [mV]', '$I_{outMOn} vs. I_{SMU}$',4)

            # 4) I Cal: DAC LIMIT vs. I Measured
            x_4,y_4,l_4 = get_and_prepare(data_IlimitvsI, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]')
            x_err_4 = np.ones_like(x_4)*3.05
            y_err_4= SMU_I_error(y_4, channel)
            # special monitoring for channel 13
            if channel == 13:
                y_4 = y_4*1000
            x_4, y_4, x_cut_4,y_cut_4, cut_4 = cut_outliers(x_4, y_4, x_err_4, y_err_4, channel)
            m_4, b_4, m_err_4, b_err_4 = plot_and_fit(x_4, y_4, x_err_4[~cut_4], y_err_4[~cut_4], x_cut_4,y_cut_4, '$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]', '$I_{lim,SMU} vs. I_{lim,DAC}$',5)
            #if channel == 13:
            #    m_4 = m_4 * 1000
            #    b_4 = b_4 * 1000

            # Calculating number of deleted points in each plot
            title = "$\\bf{Number\:of\:deleted\:points:}$\n\n"
            plot_0 = '0) $U_{out} vs. U_{DAC}$ : $\\bf{%d}$\n'%(l_0-len(x_0))
            plot_1 = '1) $U_{regulator} vs. U_{out}$ : $\\bf{%d}$\n'%(l_1-len(x_1))
            plot_2 = '2) $U_{load} vs. U_{out}$: $\\bf{%d}$\n'%(l_2-len(x_2))
            plot_3 = '3) $I_{outMON} vs. I_{SMU}$: $\\bf{%d}$\n'%(l_3-len(x_3))
            plot_4 = '4) $I_{lim,SMU} vs. I_{lim,DAC}$ : $\\bf{%d}$ \n'%(l_4-len(x_4))

            plt.figtext(0.75, 0.18,title+plot_0+plot_1+plot_2+plot_3+plot_4,bbox=dict(facecolor='lightgrey', edgecolor='red'), fontdict=None)

            deleted_points[channel][0] = l_0-len(x_0)
            deleted_points[channel][1] = l_1-len(x_1)
            deleted_points[channel][2] = l_2-len(x_2)
            deleted_points[channel][3] = l_3-len(x_3)
            deleted_points[channel][4] = l_4-len(x_4)

            # All 5 plots in one figure
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.6,
                                hspace=0.6)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Now plotting the residual PLots
            plt.subplots(figsize=(12, 6))

            # 0) Plot(U Cal: Uset vs. U out)
            residual_plots('$U_{DAC}$ [mV]', '$U_{out}$ [mV]',x_0, y_0, x_cut_0, y_cut_0, m_0, b_0,1, channel)
            # 1) U Cal: Uout vs. MonUreg
            residual_plots('$U_{out}$ [mV]', '$U_{regulator}$ [mV]',x_1, y_1, x_cut_1, y_cut_1, m_1, b_1,2, channel)
            # 2) U Cal: Uout vs. MonUload
            residual_plots('$U_{out}$ [mV]', '$U_{load}$ [mV]',x_2, y_2, x_cut_2, y_cut_2, m_2, b_2,3, channel)
            # (3) I Cal: Iout vs. IoutMon
            if (channel == 13):
                residual_plots('$I_{out(SMU)}$ [$\mu$A]', '$I_{outMon}$ [mV]',x_3, y_3, x_cut_3, y_cut_3, m_3, b_3,4, channel)
            else:
                residual_plots('$I_{out(SMU)}$ [mA]', '$I_{outMon}$ [mV]',x_3, y_3, x_cut_3, y_cut_3, m_3, b_3,4, channel)
            # 4) I Cal: DAC LIMIT vs. I Measured
            residual_plots('$I_{lim,DAC}$ [mV]', '$I_{lim,SMU}$ [mA]',x_4, y_4, x_cut_4, y_cut_4, m_4, b_4, 5, channel)

            # All 5 residual plots in one figure
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.6,
                                hspace=0.6)

            pdf.savefig()
            plt.close()

            # Now plotting the help PLots
            plt.subplots(figsize=(12, 6))
            # 1) I Cal: Iout vs. UoutMon- horizontal line
            help_plots(data_IvsI, '$I_{out(SMU)}$ [mA]', '$U_{outMon}$', 'I Cal: Iout vs. UoutMon - horizontal line', 1)
            # 2) I Cal: Iout vs. U-SMU - horizontal line
            help_plots(data_IvsI, '$I_{out(SMU)}$ [mA]', '$U_{SMU}$', 'I Cal: Iout vs. U-SMU - horizontal line', 2)
            # 3) I Cal: Iout vs. StatBit - should be high
            help_plots(data_IvsI, '$I_{out(SMU)}$ [mA]', 'StatBit', 'I Cal: Iout vs. StatBit - should be high', 3)
            # 4) Limit Cal: I Limit out vs. StatBit - should be low
            help_plots(data_IlimitvsI, '$I_{lim,SMU}$ [mA]', 'StatBit', 'LimitCal: I Limit vs. StatBit - should be low',4)

            # All 5 residual plots in one figure
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.6,
                                hspace=0.6)

            pdf.savefig()
            plt.close()

            config_ini['Information'] = {'date': m_time,
                                    'format': 'yyyy-mm-dd'}
            write_in_ini(config_ini, channel, m_0, b_0, m_1, b_1, m_2, b_2, m_3, b_3, m_4, b_4)
            write_in_ini(config_err, channel, m_err_0, b_err_0, m_err_1, b_err_1, m_err_2, b_err_2, m_err_3, b_err_3, m_err_4, b_err_4)

            print(f'Calculations for Channel {channel} finished')

        with open(os.path.join(config["calibration_data"].get("data_path"),'constants.ini'), 'w') as configfile:
            config_ini.write(configfile)
        with open(os.path.join(config["calibration_data"].get("data_path"),'constants_err.ini'), 'w') as configfile:
            config_err.write(configfile)

    print('Plotting Histogram with Number of deleted points...')
    histo_deleted_points(l_1,deleted_points)
    print('Checking if Calibration was successful...\n')
    success = pass_fail(l_1,deleted_points)
    path = config['calibration_data'].get('data_path')
    if __name__ == '__main__':
        if success == True:
            add_to_database(path)
    return success

if __name__ == '__main__':
    main()