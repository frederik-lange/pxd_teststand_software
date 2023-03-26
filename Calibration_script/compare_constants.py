"""
This program compares the calibration constants from two constants.ini files
It displays the differences visually in the Compare_constants.pdf file
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from matplotlib.backends.backend_pdf import PdfPages
import csv

ps = 105

path1 = f'../data'
#path1 = f'../data/CalibrationData/ps{ps}/PS_{ps}_20230225'
path2 = f'../data/CalibrationData/ps{ps}/PS_{ps}_20230225_60points'
pdf_path = f'../data/CalibrationData/ps{ps}/PS_{ps}_compare_60_points_to_range_1.pdf'
delta_gains, delta_offsets, std_gains, std_offsets = np.zeros(24), np.zeros(24), np.zeros(24), np.zeros(24)

def gain_diff(gain1, gain2):
    return (gain2-gain1)/gain1*100

def offset_diff(offset1, offset2):
    return offset2-offset1

def compare_2_cals():
    config1 = configparser.ConfigParser()
    config2 = configparser.ConfigParser()
    config1.read(os.path.join(path1,'constants.ini')), config2.read(os.path.join(path2,'constants.ini'))
    with PdfPages(pdf_path) as pdf:
        for name in ['DAC_VOLTAGE','ADC_U_LOAD','ADC_U_REGULATOR','ADC_I_MON','DAC_CURRENT']:
            print(f"Plotting {name}...")
            for channel in range(24):
                gains1, gains2 = float(config1[f'{channel}'][name+'_GAIN']), float(config2[f'{channel}'][name+'_GAIN'])
                delta_gains[channel] = gain_diff(gains1,gains2)
                offsets1, offsets2 = float(config1[f'{channel}'][name+'_OFFSET']), float(config2[f'{channel}'][name+'_OFFSET'])
                delta_offsets[channel] = offset_diff(offsets1, offsets2)

            # Plot Gains
            plt.figure()
            plt.grid()
            plt.xticks(range(24))
            plt.axhline(0, color='black')
            plt.bar(np.arange(0, 24, 1), delta_gains, color='r')
            plt.title(name + ' Gain')
            plt.xlabel('Channel'), plt.ylabel('Relative difference in %')
            pdf.savefig()
            plt.close()
            # Plot Offsets
            plt.figure()
            plt.grid()
            plt.xticks(range(24))
            plt.axhline(0, color='black')
            plt.bar(np.arange(0, 24, 1), delta_offsets, color='r')
            plt.title(name + ' Offset')
            plt.xlabel('Channel'), plt.ylabel('Absolute difference')
            pdf.savefig()
            plt.close()
            print(name,"Relative gain difference:\n",delta_gains)
            print(name,"Absolute offset difference:\n",delta_offsets)

def compare_to_average():
    config_avg = configparser.ConfigParser()
    config_cal = configparser.ConfigParser()
    config_std = configparser.ConfigParser()
    config_avg.read(os.path.join(path1, f'PS_{ps}_constants.ini')), config_cal.read(os.path.join(path2, 'constants.ini')), config_std.read(os.path.join(path1,f'PS_{ps}_constants_std.ini'))
    with PdfPages(pdf_path) as pdf:
        for name in ['DAC_VOLTAGE', 'ADC_U_LOAD', 'ADC_U_REGULATOR', 'ADC_I_MON', 'DAC_CURRENT']:
            print(f"Plotting {name}...")
            for channel in range(24):
                gains1, gains2 = float(config_avg[f'{channel}'][name + f'_GAIN_{channel}']), float(config_cal[f'{channel}'][name + '_GAIN'])
                delta_gains[channel] = gain_diff(gains1, gains2)
                offsets1, offsets2 = float(config_avg[f'{channel}'][name + f'_OFFSET_{channel}']), float(config_cal[f'{channel}'][name + '_OFFSET'])
                delta_offsets[channel] = offset_diff(offsets1, offsets2)
                std_gains[channel], std_offsets[channel] = float(config_std[f'{channel}'][name + f'_GAIN_{channel}'])/gains1, float(config_std[f'{channel}'][name + f'_OFFSET_{channel}'])
            # Plot Gains

            x = np.arange(0,24,1)
            sigma_env = 1
            width = 1
            plt.bar(x, sigma_env*std_gains, color='b', alpha=0.5,width = width)
            plt.bar(x, -sigma_env*std_gains, color='b', alpha=0.5,width = width, label='standard deviation')
            plt.bar(x, delta_gains, color='r', alpha =0.7,width = width, label='difference')
            plt.title(name + ' Gain')
            plt.xlabel('Channel'), plt.ylabel('Relative difference in %')
            plt.legend()
            pdf.savefig()
            plt.close()
            # Plot Offsets
            plt.figure()
            plt.grid()
            plt.xticks(range(24))
            plt.axhline(0, color='black')
            plt.bar(x, sigma_env*std_offsets, color='b', alpha=0.5,width = width)
            plt.bar(x, -sigma_env*std_offsets, color='b', alpha=0.5,width = width, label='standard deviation')
            plt.bar(x, delta_offsets, color='r', alpha =0.7,width = width, label='difference')
            plt.title(name + ' Offset')
            plt.xlabel('Channel'), plt.ylabel('Absolute difference')
            plt.legend()
            pdf.savefig()
            plt.close()
            print(name, "Relative gain difference:\n", delta_gains)
            print(name, "Absolute offset difference:\n", delta_offsets)

def max_differences():
    config_avg = configparser.ConfigParser()
    config_cal = configparser.ConfigParser()
    config_std = configparser.ConfigParser()
    config_avg.read(os.path.join(path1, f'PS_{ps}_constants.ini'))
    config_std.read(os.path.join(path1, f'PS_{ps}_constants_std.ini'))
    names = ['DAC_VOLTAGE', 'ADC_U_LOAD', 'ADC_U_REGULATOR', 'ADC_I_MON', 'DAC_CURRENT']
    calibrations = ['PS_105_40points','PS_105_50points','PS_105_60points','PS_105_60points_2','PS_105_70points']
    max_diff = np.zeros((240,5))
    for cal in range(5):
        config_cal.read(os.path.join('../data/CalibrationData/ps105',calibrations[cal], 'constants.ini'))
        print(os.path.join('../data/CalibrationData/ps105',calibrations[cal], 'constants.ini'))
        for channel in range(24):
            for var in range(5):
                avg_gains, cal_gains = float(config_avg[f'{channel}'][names[var] + f'_GAIN_{channel}']), float(config_cal[f'{channel}'][names[var] + '_GAIN'])
                max_diff[channel*10+var*2,cal] = gain_diff(avg_gains,cal_gains)
                avg_offsets, cal_offsets =float(config_avg[f'{channel}'][names[var] + f'_OFFSET_{channel}']), float(config_cal[f'{channel}'][names[var] + '_OFFSET'])
                max_diff[channel*10+var*2+1,cal] = offset_diff(cal_offsets, avg_offsets)
    print(max_diff.shape)
    with open('../data/PS_105_max_differences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(max_diff)
                #std_gains[channel], std_offsets[channel] = float(config_std[f'{channel}'][name + f'_GAIN_{channel}']) / gains1, float(config_std[f'{channel}'][name + f'_OFFSET_{channel}'])

def calibration_optimization():
    data = np.loadtxt('../data/PS_105_max_differences.csv', delimiter=',', skiprows=0)
    config_vals = configparser.ConfigParser()
    config_vals.read('../data/PS_105_constants.ini')
    config_stds = configparser.ConfigParser()
    config_stds.read('../data/PS_105_constants_std.ini')
    names = ['DAC_VOLTAGE', 'ADC_U_LOAD', 'ADC_U_REGULATOR', 'ADC_I_MON', 'DAC_CURRENT']
    calibrations = ['40 points', '50 points', '60 points', '60 points', '70 points']
    colors = ['b', 'r', 'g', 'm', 'c']
    with PdfPages('../data/different_points.pdf') as pdf:
        for var in range(5):
            plt.figure()
            plt.grid()
            x = np.arange(0, 24, 1)
            plt.xticks(x)
            plt.axhline(0, color='black')
            # Gains:
            for channel in range(24):
                std_rel =  float(config_stds[f'{channel}'][names[var]+f'_GAIN_{channel}']) / float(config_vals[f'{channel}'][names[var]+f'_GAIN_{channel}']) * 100*3
                if channel == 0:
                    plt.bar(channel + 0.5, 2 * std_rel, bottom=-std_rel, width=1, alpha=0.5, color='grey', label='$3\sigma$')
                else:
                    plt.bar(channel + 0.5, 2 * std_rel, bottom=-std_rel, width=1, alpha=0.5, color='grey')
                for cal in range(5):
                    if channel == 0:
                        plt.bar(channel+0.2*cal+0.1,data[channel*10+var*2,cal],label=f'{calibrations[cal]}',width=0.2, color=colors[cal],alpha = 1)
                    else:
                        plt.bar(channel + 0.2 * cal+0.1, data[channel * 10 + var * 2, cal], width=0.2, color=colors[cal],alpha = 1)
            plt.title(names[var] + ' GAIN')
            plt.ylabel('Relative difference in %'), plt.xlabel('Channels')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Offsets:
            plt.figure()
            plt.grid()
            x = np.arange(0, 24, 1)
            plt.xticks(x)
            plt.axhline(0, color='black')
            # Gains:
            for channel in range(24):
                diff_abs = float(config_stds[f'{channel}'][names[var] + f'_OFFSET_{channel}'])*3
                if channel == 0:
                    plt.bar(channel + 0.5, 2 * diff_abs, bottom=-diff_abs, width=1, alpha=0.5, color='grey',label='$3\sigma$')
                else:
                    plt.bar(channel + 0.5, 2 * diff_abs, bottom=-diff_abs, width=1, alpha=0.5, color='grey')
                for cal in range(5):
                    if channel == 0:
                        plt.bar(channel+0.2*cal+0.1,data[channel*10+var*2+1,cal],label=f'{calibrations[cal]}',width=0.2, color=colors[cal],alpha = 1)
                    else:
                        plt.bar(channel + 0.2 * cal+0.1, data[channel * 10 + var * 2+1, cal], width=0.2, color=colors[cal],alpha = 1)
            plt.title(names[var] + ' Offset')
            plt.ylabel('Absolute difference'), plt.xlabel('Channels')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    calibration_optimization()
    '''for x in range(24):
        for y in range(1,4):
            sx, sy = str(x), str(y)
            if y == 3:
                print("\\begin{figure}\n \\centering\n \\includegraphics[page=",sy,",angle=90,height=0.95\\textheight]{figs/appendix/Channel_",sx,".pdf}\n\\caption{PS 105 Channel ",sx,"}\n \\label{fig:ps105_ch",sx,"}\n\end{figure}",sep='')
                print("\\clearpage")
            else:
                print("\\begin{figure}\n \\centering\n \\includegraphics[page=", sy,",angle=90,height=0.95\\textheight]{figs/appendix/Channel_", sx, ".pdf}\n\end{figure}", sep='')
    '''