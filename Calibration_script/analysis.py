import os
import configparser
from Calibration_script import main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

names = ["Unit","Date","validated","used_for_range"]
for i in range(24):
    names.append(f"DAC_VOLTAGE_GAIN_{i}")
    names.append(f"DAC_VOLTAGE_OFFSET_{i}")
    names.append(f"ADC_U_LOAD_GAIN_{i}")
    names.append(f"ADC_U_LOAD_OFFSET_{i}")
    names.append(f"ADC_U_REGULATOR_GAIN_{i}")
    names.append(f"ADC_U_REGULATOR_OFFSET_{i}")
    names.append(f"ADC_I_MON_GAIN_{i}")
    names.append(f"ADC_I_MON_OFFSET_{i}")
    names.append(f"DAC_CURRENT_GAIN_{i}")
    names.append(f"DAC_CURRENT_OFFSET_{i}")

channel_names = ['(dhp-io)',
'(sw-dvdd)',
'(dcd-dvdd)',
'(dhp-core)',
'(dcd-refin)',
'(source)',
'(dcd-avdd)',
'(amplow)',
'(ccg1)',
'(ccg2)',
'(drift)',
'(ccg3)',
'(poly)',
'(HV)',
'(guard)',
'(bulk)',
'(gate-on1)',
'(gate-on2)',
'(gate-off)',
'(gate-on3)',
'(clear-on)',
'(sw-refin)',
'(sw-sub)',
'(clear-off)']

def check_all():
    configPath = configparser.ConfigParser()
    config = configparser.ConfigParser()
    config_ini = configparser.ConfigParser()
    configPath.read('path.ini')
    scans, successes = 0,0
    for i in range(20,110):
        print(f"Power supply {i}")
        for root, dirs, files in os.walk(f'../data/CalibrationData/ps{i}'):
            for dir in dirs:
                if os.path.exists(os.path.join(root,dir,'Channel_0_U_vs_U.dat')):
                    scans += 1
                    configPath['calibration_data']['data_path'] = os.path.join(root,dir)
                    with open('path.ini', 'w') as configPathfile:
                        configPath.write(configPathfile)
                    config_ini.read(os.path.join(configPath['calibration_data'].get('data_path'),'constants.ini'))
                    print(config_ini['Information']['success'],config_ini['Information'].get('date'))
                    wrong_channels = np.zeros(24)
                    for channel in range(24):
                        for name in ['DAC_VOLTAGE_GAIN','DAC_VOLTAGE_OFFSET','ADC_U_LOAD_GAIN','ADC_U_LOAD_OFFSET','ADC_U_REGULATOR_GAIN','ADC_U_REGULATOR_OFFSET',
                                     'ADC_I_MON_GAIN','ADC_I_MON_OFFSET','DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET']:
                            if main.check_range(name,channel,config_ini) == False:
                                wrong_channels[channel] = 1
                    if np.sum(wrong_channels) > 0:
                        print('\nCalibration was NOT successful! Please check warnings!')
                        success = False
                    else:
                        print('\nCalibration was successful!')
                        success = True
                        successes += 1

                    # write calibration result in ini file
                    with open(os.path.join(configPath["calibration_data"].get("data_path"), 'constants.ini'),
                              'w') as configfile:
                        if success == True:
                            config_ini["Information"]["success"] = "True"
                        else:
                            config_ini["Information"]["success"] = "False"
                        config_ini.write(configfile)
    print(f'{successes} out of {scans} data sets were successful!')

def boxplot_per_channel():
    data = pd.read_csv(f'./../data/database.csv')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/database_boxplots_per_channel.pdf') as pdf:
        for channel in range(24):
            print(f"Plotting Channel {channel}...")
            y = np.zeros((10,len(data[names[0]])))
            list = []
            for n in range(10):
                y[n] = data[names[4 + n*24 + channel]]
                list.append(y[n])
            fig,ax = plt.subplots(2,5)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            fig.suptitle(f'Channel {channel}')
            for n in range(10):
                if n <= 4:
                    a = 0
                else:
                    a = 1
                plt.grid()
                ax[a,n%5].boxplot(y[n])
                ax[a,n%5].set_title(plotnames[n])
            plt.tight_layout()
            plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9)
            pdf.savefig()
            plt.close(fig)

def boxplot_per_constant():
    data = pd.read_csv(f'./../data/database.csv')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/database_boxplots_per_constant.pdf') as pdf:
        x = np.arange(0, 24, 1)
        print(x)
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            mask = data['used_for_range'] == 'yes'
            fig, ax = plt.subplots()
            y = np.zeros((24, len(data[names[0]][mask])))
            list = []
            for channel in range(24):
                y[channel] = data[names[4 + channel*10 + n]][mask]
                list.append(y[channel])
            plt.xlabel('Channels')
            plt.title(plotnames[n])
            ax.boxplot(list, positions=x)
            plt.xticks(x)
            pdf.savefig()
            plt.close(fig)

def constant_precision():
    pass

def channel_grouping_by_board():
    #data = pd.read_csv(f'./../data/database.csv')
    # mask = data['used_for_range'] == 'yes'
    config_med, config_std = configparser.ConfigParser(), configparser.ConfigParser()
    config_med.read('../data/database.ini')
    config_std.read('../data/database_std.ini')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/channel_grouping_by_board.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            plt.title(plotnames[n])
            fig, ax = plt.subplots(2,3)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            fig.suptitle(plotnames[n])
            medians, stds = np.zeros(24), np.zeros(24)
            for channel in range(24):
                medians[channel] = float(config_med[f'{channel}'][plotnames[n]+f'_{channel}'])
                stds[channel] = float(config_std[f'{channel}'][plotnames[n]+f'_{channel}'])
            for plot in range(6):
                x = np.arange(plot*4,plot*4+4,1)
                xt = x
                for i in xt:
                    i = str(i)
                x = np.arange(0, len(x), 1)
                y_med = medians[plot*4:plot*4+4]
                y_std = stds[plot*4:plot*4+4]
                if plot<3:
                    ax[0,plot%3].bar(x,2*y_std,bottom=y_med-y_std,alpha=1)
                    ax[0, plot % 3].hlines(y_med,x-0.5,x+0.5)
                    ax[0, plot % 3].set_xticks(x)
                    ax[0, plot % 3].set_xticklabels(xt)
                else:
                    ax[1, plot % 3].bar(x, 2 * y_std, bottom=y_med - y_std,alpha=1)
                    ax[1, plot % 3].hlines(y_med, x - 0.5, x + 0.5)
                    ax[1, plot % 3].set_xticks(x)
                    ax[1, plot % 3].set_xticklabels(xt)
                plt.xticks(x)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

def channel_grouping():
    config_med, config_std = configparser.ConfigParser(), configparser.ConfigParser()
    config_med.read('../data/database.ini')
    config_std.read('../data/database_std.ini')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    group1 = [0,1,2,3,4,6,7]
    group2 = [8,9,10,11]
    group3 = [16,17,18,19]
    group4 = [20,23]
    group5 = [21,22]
    group6 = [5,12,13,14,15]
    groups = [group1, group2, group3, group4, group5, group6]
    with PdfPages(f'../data/channel_grouping.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            plt.title(plotnames[n])
            fig, ax = plt.subplots(2, 3)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            fig.suptitle(plotnames[n])
            medians, stds = np.zeros(24), np.zeros(24)
            for channel in range(24):
                medians[channel] = float(config_med[f'{channel}'][plotnames[n] + f'_{channel}'])
                stds[channel] = float(config_std[f'{channel}'][plotnames[n] + f'_{channel}'])
            for plot in range(6):
                x = np.array(groups[plot])
                xt = x
                for i in xt:
                    i = str(i)
                x = np.arange(0,len(x),1)
                #y = np.array
                y_med = medians[np.array(groups[plot])]
                y_std = stds[np.array(groups[plot])]
                if plot < 3:
                    ax[0, plot % 3].bar(x, 2 * y_std, bottom=y_med - y_std, alpha=1)
                    ax[0, plot % 3].hlines(y_med, x - 0.5, x + 0.5)
                    ax[0, plot % 3].set_xticks(x)
                    ax[0, plot % 3].set_xticklabels(xt)
                else:
                    ax[1, plot % 3].bar(x, 2 * y_std, bottom=y_med - y_std, alpha=1)
                    ax[1, plot % 3].hlines(y_med, x - 0.5, x + 0.5)
                    ax[1, plot % 3].set_xticks(x)
                    ax[1, plot % 3].set_xticklabels(xt)
                #plt.xticks(xt)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

def grouping_boxplots():
    data = pd.read_csv('./../data/database.csv')
    mask = data['used_for_range'] == 'yes'
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    group1 = [0, 1, 2, 3, 4, 6, 7]
    group2 = [8, 9, 10, 11]
    group3 = [16, 17, 18, 19]
    group4 = [20, 23]
    group5 = [21, 22]
    group6 = [5, 12, 13, 14, 15]
    groups = [group1, group2, group3, group4, group5, group6]
    with PdfPages(f'../data/channel_grouping_boxplots.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            fig, ax = plt.subplots(2, 3)
            fig.set_figheight(5)
            fig.set_figwidth(12)
            st = fig.suptitle(plotnames[n])
            for plot in range(6):
                x = np.array(groups[plot])
                xt = []
                for i in range(len(x)):
                    #print(channel_names[int(xt[i])])
                    s1 = str(x[i])
                    s2 = channel_names[int(x[i])]
                    if plot == 0: # or plot == 5:
                        if i%2 == 1:
                            xt.append(''.join((s1, "\n\n", s2)))
                        else:
                            xt.append(''.join((s1, "\n", s2)))
                    else:
                        xt.append(''.join((s1, "\n", s2)))
                y = np.zeros(len(data[names[0]][mask]))
                list = []
                for g in groups[plot]:
                    arg = int(float(g))
                    y = data[names[4 + arg * 10 + n]][mask]
                    list.append(y)
                if plot < 3:
                    ax[0, plot % 3].boxplot(list)
                    ax[0, plot % 3].set_xticklabels(xt)
                    #ax2 = ax[0, plot%3].twiny()
                    #ax2.set_xticklabels(x)
                else:
                    ax[1, plot % 3].boxplot(list)
                    ax[1, plot % 3].set_xticklabels(xt)
                    #ax2 = ax[1, plot%3].twiny()
                    #ax2.set_xticklabels(x)
                # secret techniques
                #plt.setp(ax[0,0].get_xticklabels(), rotation=90, horizontalalignment='center')
                #plt.setp(ax[0, 2].get_xticklabels(), rotation=90, horizontalalignment='center')
                #plt.setp(ax[1, 2].get_xticklabels(), rotation=90, horizontalalignment='center')
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    #boxplot_per_constant()
    #boxplot_per_channel()
    #channel_grouping()
    #channel_grouping_by_board()
    grouping_boxplots()