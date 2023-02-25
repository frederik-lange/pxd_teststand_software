"""
This program compares the calibration constants from two constants.ini files
It displays the differences visually in the Compare_constants.pdf file
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from matplotlib.backends.backend_pdf import PdfPages

ps = 105

path1 = f'../data'
#path1 = f'../data/CalibrationData/ps{ps}/PS_{ps}_20230225'
path2 = f'../data/CalibrationData/ps{ps}/PS_{ps}_20230223_40points'
pdf_path = f'../data/CalibrationData/ps{ps}/PS_{ps}_compare_40_points_to_range.pdf'
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
                gains1, gains2 = float(config_avg[f'{channel}'][name + '_GAIN']), float(config_cal[f'{channel}'][name + '_GAIN'])
                delta_gains[channel] = gain_diff(gains1, gains2)
                offsets1, offsets2 = float(config_avg[f'{channel}'][name + '_OFFSET']), float(config_cal[f'{channel}'][name + '_OFFSET'])
                delta_offsets[channel] = offset_diff(offsets1, offsets2)
                std_gains[channel], std_offsets[channel] = float(config_std[f'{channel}'][name + '_GAIN'])/gains1, float(config_std[f'{channel}'][name + '_OFFSET'])
            # Plot Gains
            plt.figure()
            plt.grid()
            plt.xticks(range(24))
            plt.axhline(0, color='black')
            plt.bar(np.arange(0, 24, 1), std_gains, color='b', alpha=0.5)
            plt.bar(np.arange(0, 24, 1), -std_gains, color='b', alpha=0.5)
            plt.bar(np.arange(0, 24, 1), delta_gains, color='r', alpha =0.7)
            plt.title(name + ' Gain')
            plt.xlabel('Channel'), plt.ylabel('Relative difference in %')
            pdf.savefig()
            plt.close()
            # Plot Offsets
            plt.figure()
            plt.grid()
            plt.xticks(range(24))
            plt.axhline(0, color='black')
            plt.bar(np.arange(0, 24, 1), std_offsets, color='b', alpha=0.5)
            plt.bar(np.arange(0, 24, 1), -std_offsets, color='b', alpha=0.5)
            plt.bar(np.arange(0, 24, 1), delta_offsets, color='r', alpha =0.7)
            plt.title(name + ' Offset')
            plt.xlabel('Channel'), plt.ylabel('Absolute difference')
            pdf.savefig()
            plt.close()
            print(name, "Relative gain difference:\n", delta_gains)
            print(name, "Absolute offset difference:\n", delta_offsets)

if __name__ == '__main__':
    compare_to_average()