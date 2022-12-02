"""
    Part of the multi calibration
    This program can compare the variation of constants over multiple calibration runs.
    E.g. it can display the change of constants over time
    Output: Calibration_constants_ps.pdf and Calibration_constants_ps_var.pdf in folder 'ps(number)'
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import numpy as np



ps = 15
path = f'../data/ps{ps}/'
values = f'../data/ps{ps}/output.csv'
errors = f'../data/ps{ps}/output_err.csv'
path_pdf = f'../data/ps{ps}/Calibration_constants_ps{ps}.pdf'
path_var = f'../data/ps{ps}/Calibration_constants_ps{ps}_var.pdf'

def plot_data(channel):
    i = 10 * channel + channel
    with open(values, 'r') as csvfile:
        constants = csv.reader(csvfile, delimiter=',')
        rows = list(constants)
        #run = rows[0] # plot against date
        run = np.arange(0,21) # plot against calibration number
        DAC_VOLTAGE_GAIN = rows[i + 2]
        DAC_VOLTAGE_OFFSET = rows[i + 3]
        ADC_U_LOAD_GAIN = rows[i + 4]
        ADC_U_LOAD_OFFSET = rows[i + 5]
        ADC_U_REGULATOR_GAIN = rows[i + 6]
        ADC_U_REGULATOR_OFFSET = rows[i + 7]
        ADC_I_MON_GAIN = rows[i + 8]
        ADC_I_MON_OFFSET = rows[i + 9]
        DAC_CURRENT_GAIN = rows[i + 10]
        DAC_CURRENT_OFFSET = rows[i + 11]

        with open(errors, 'r') as csvfile:
            constants = csv.reader(csvfile, delimiter=',')
            rows = list(constants)
            DAC_VOLTAGE_GAIN_ERROR = rows[i + 1]
            DAC_VOLTAGE_OFFSET_ERROR = rows[i + 2]
            ADC_U_LOAD_GAIN_ERROR = rows[i + 3]
            ADC_U_LOAD_OFFSET_ERROR = rows[i + 4]
            ADC_U_REGULATOR_GAIN_ERROR = rows[i + 5]
            ADC_U_REGULATOR_OFFSET_ERROR = rows[i + 6]
            ADC_I_MON_GAIN_ERROR = rows[i + 7]
            ADC_I_MON_OFFSET_ERROR = rows[i + 8]
            DAC_CURRENT_GAIN_ERROR = rows[i + 9]
            DAC_CURRENT_OFFSET_ERROR = rows[i + 10]

            var_g_1, abw_g_1 = plot(run[1:],DAC_VOLTAGE_GAIN[1:],DAC_VOLTAGE_GAIN_ERROR[1:], 'mV/mV', 'lightcoral', 'DAC_VOLTAGE_GAIN', 1)
            var_o_1, abw_o_1 = plot(run[1:], DAC_VOLTAGE_OFFSET[1:],DAC_VOLTAGE_OFFSET_ERROR[1:], 'mV', 'darkgrey', 'DAC_VOLTAGE_OFFSET', 2)
            var_g_2, abw_g_2 = plot(run[1:], ADC_U_LOAD_GAIN[1:],ADC_U_LOAD_GAIN_ERROR[1:], 'mV/mV', 'orange', 'ADC_U_LOAD_GAIN', 3)
            var_o_2, abw_o_2 = plot(run[1:], ADC_U_LOAD_OFFSET[1:],ADC_U_LOAD_OFFSET_ERROR[1:], 'mV', 'darkseagreen', 'ADC_U_LOAD_OFFSET', 4)
            var_g_3, abw_g_3 = plot(run[1:], ADC_U_REGULATOR_GAIN[1:],ADC_U_REGULATOR_GAIN_ERROR[1:], 'mV/mV', 'cadetblue', 'ADC_U_REGULATOR_GAIN', 5)
            var_o_3, abw_o_3 = plot(run[1:], ADC_U_REGULATOR_OFFSET[1:],ADC_U_REGULATOR_OFFSET_ERROR[1:], 'mV', 'cornflowerblue', 'ADC_U_REGULATOR_OFFSET', 6)
            var_g_4, abw_g_4 = plot(run[1:], ADC_I_MON_GAIN[1:],ADC_I_MON_GAIN_ERROR[1:], 'mA/mV', 'slateblue', 'ADC_I_MON_GAIN', 7)
            var_o_4, abw_o_4 = plot(run[1:], ADC_I_MON_OFFSET[1:],ADC_I_MON_OFFSET_ERROR[1:], 'mV', 'thistle', 'ADC_I_MON_OFFSET', 8)
            var_g_5, abw_g_5 = plot(run[1:], DAC_CURRENT_GAIN[1:],DAC_CURRENT_GAIN_ERROR[1:], 'mA/mV', 'plum', 'DAC_CURRENT_GAIN', 9)
            var_o_5, abw_o_5 = plot(run[1:], DAC_CURRENT_OFFSET[1:],DAC_CURRENT_OFFSET_ERROR[1:], 'mA', 'palevioletred', 'DAC_CURRENT_OFFSET', 10)

            return(var_g_1, var_o_1, var_g_2,var_o_2, var_g_3, var_o_3, var_g_4, var_o_4, var_g_5, var_o_5, abw_g_1,abw_o_1,abw_g_2,abw_o_2,abw_g_3,abw_o_3,abw_g_4,abw_o_4,abw_g_5,abw_o_5)


def plot(x_data, y_data,err, y_ax, color, title, n):
    plt.subplot(10, 1, n)
    y = [eval(i) for i in y_data]
    y_err = [eval(i) for i in err]
    mean = np.mean(y[1:])
    std = np.std(y[1:])
    if mean == 0:
        var = 0
    else:
        var = abs(std / mean)

    if (y[0]) == 0 or mean == 0:
        abw = 0
    else:
        abw = ((y[0] - mean) / mean) * 100

    stdb = std+std
    plotmean = []
    for i in range(len(y)):
        plotmean.append(mean)

    plt.scatter(x_data, y, color=color, linewidths=7.0, label=title)
    plt.plot(x_data,plotmean,'-',color=color,linewidth=3.0,label='median')
    plt.fill_between(x_data,plotmean+stdb,plotmean-stdb,  alpha=0.2, color=color, label='2$\sigma$')
    plt.errorbar(x_data, y, yerr=y_err, fmt='none', ecolor='grey')
    plt.xticks(fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(y_ax, fontsize =14)
    plt.legend(prop={'size': 15})

    return (var, abw)


def plot_histo(x,y,title,color):

    plt.bar(x, y, color=color, label=title)
    plt.xlabel('Channel')
    plt.xticks(np.arange(0, 24, 1))
    plt.ylabel('Variation coefficient')
    plt.legend(prop={'size': 8})

def plot_histo_abw(x,y,title,color):

    plt.bar(x, y, color=color, label=title)
    plt.xlabel('Channel')
    plt.xticks(np.arange(0, 24, 1))
    plt.ylabel('Relative difference of constant to mean [%]')
    plt.legend(prop={'size': 8})

def main():
    with PdfPages(path_pdf) as pdf:
        channel_histo = []
        for i in range(24):
            channel_histo.append(i)
        l_var_g_1 = []
        l_var_o_1 = []
        l_var_g_2 = []
        l_var_o_2 = []
        l_var_g_3 = []
        l_var_o_3 = []
        l_var_g_4 = []
        l_var_o_4 = []
        l_var_g_5 = []
        l_var_o_5 = []

        ol_var_g_1 = []
        ol_var_o_1 = []
        ol_var_g_2 = []
        ol_var_o_2 = []
        ol_var_g_3 = []
        ol_var_o_3 = []
        ol_var_g_4 = []
        ol_var_o_4 = []
        ol_var_g_5 = []
        ol_var_o_5 = []

        for channel in range(24):
            print('Plotting Channel %d' %channel)
            plt.subplots(figsize=(20, 30))
            (var_g_1, var_o_1, var_g_2,var_o_2, var_g_3, var_o_3, var_g_4, var_o_4, var_g_5, var_o_5, abw_g_1,abw_o_1,abw_g_2,abw_o_2,abw_g_3,abw_o_3,abw_g_4,abw_o_4,abw_g_5,abw_o_5) = plot_data(channel)
            plt.suptitle('Channel %d\n'%channel, size=40.0)
            plt.xlabel('\n Date of Calibration Run', size=25.0)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            l_var_g_1.append(var_g_1)
            l_var_o_1.append(var_o_1)
            l_var_g_2.append(var_g_2)
            l_var_o_2.append(var_o_2)
            l_var_g_3.append(var_g_3)
            l_var_o_3.append(var_o_3)
            l_var_g_4.append(var_g_4)
            l_var_o_4.append(var_o_4)
            l_var_g_5.append(var_g_5)
            l_var_o_5.append(var_o_5)

            ol_var_g_1.append(abw_g_1)
            ol_var_o_1.append(abw_o_1)
            ol_var_g_2.append(abw_g_2)
            ol_var_o_2.append(abw_o_2)
            ol_var_g_3.append(abw_g_3)
            ol_var_o_3.append(abw_o_3)
            ol_var_g_4.append(abw_g_4)
            ol_var_o_4.append(abw_o_4)
            ol_var_g_5.append(abw_g_5)
            ol_var_o_5.append(abw_o_5)



        with PdfPages(path_var) as pdf:
            plot_histo(channel_histo, l_var_g_1,  'DAC_VOLTAGE_GAIN', 'lightcoral')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_o_1,  'DAC_VOLTAGE_OFFSET', 'darkgrey')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_g_2, 'ADC_U_LOAD_GAIN', 'orange')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_o_2, 'ADC_U_LOAD_OFFSET', 'darkseagreen')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_g_3, 'ADC_U_REGULATOR_GAIN', 'cadetblue')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_o_3, 'ADC_U_REGULATOR_OFFSET', 'cornflowerblue')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_g_4, 'ADC_I_MON_GAIN', 'slateblue')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_o_4, 'ADC_I_MON_OFFSET', 'thistle')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_g_5, 'DAC_CURRENT_GAIN', 'plum')
            pdf.savefig()
            plt.close()
            plot_histo(channel_histo, l_var_o_5, 'DAC_CURRENT_OFFSET', 'palevioletred')
            pdf.savefig()
            plt.close()

        #with PdfPages(path_old) as pdf:
        #    plot_histo_abw(channel_histo, ol_var_g_1,  'DAC_VOLTAGE_GAIN', 'lightcoral')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_o_1,  'DAC_VOLTAGE_OFFSET', 'darkgrey')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_g_2, 'ADC_U_LOAD_GAIN', 'orange')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_o_2, 'ADC_U_LOAD_OFFSET', 'darkseagreen')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_g_3, 'ADC_U_REGULATOR_GAIN', 'cadetblue')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_o_3, 'ADC_U_REGULATOR_OFFSET', 'cornflowerblue')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_g_4, 'ADC_I_MON_GAIN', 'slateblue')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_o_4, 'ADC_I_MON_OFFSET', 'thistle')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_g_5, 'DAC_CURRENT_GAIN', 'plum')
        #    pdf.savefig()
        #    plt.close()
        #    plot_histo_abw(channel_histo, ol_var_o_5, 'DAC_CURRENT_OFFSET', 'palevioletred')
        #    pdf.savefig()
        #    plt.close()
        #    print('done')



if __name__ == '__main__':
    main()