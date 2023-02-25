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
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            mask = data['used_for_range'] == 'yes'
            fig, ax = plt.subplots()
            x = np.arange(0, 24, 1)
            y = np.zeros((24, len(data[names[0]][mask])))
            list = []
            for channel in range(24):
                y[channel] = data[names[4 + channel*10 + n]][mask]
                list.append(y[channel])
            plt.xlabel('Channels')
            plt.title(plotnames[n])
            ax.boxplot(list)
            pdf.savefig()
            plt.close(fig)

def constant_precision():


if __name__ == '__main__':
    boxplot_per_constant()
    boxplot_per_channel()