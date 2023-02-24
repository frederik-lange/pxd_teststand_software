"""
This program compares the calibration constants from two constants.ini files
It displays the differences visually in the Compare_constants.pdf file
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from matplotlib.backends.backend_pdf import PdfPages

path1 = '../data/CalibrationData/ps83/PS_83_20230222'
path2 = '../data/CalibrationData/ps83/PS_83_20230224'
pdf_path = '../data/CalibrationData/ps83/PS_83_compare_new_to_old.pdf'
delta_gains, delta_offsets = np.zeros(24), np.zeros(24)

def gain_diff(gain1, gain2):
    return (gain2-gain1)/gain1*100

def offset_diff(offset1, offset2):
    return offset2-offset1

def main():
    config1 = configparser.ConfigParser()
    config2 = configparser.ConfigParser()
    config1.read(os.path.join(path1,'constants.ini')), config2.read(os.path.join(path2,'constants.ini'))
    with PdfPages(pdf_path) as pdf:
        for name in ['DAC_VOLTAGE','ADC_U_LOAD','ADC_U_REGULATOR','ADC_I_MON','DAC_CURRENT']:
            print(f"Plotting {name}...")
            for channel in range(24):
                #print(f"Plotting Channel {channel}...")
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

if __name__ == '__main__':
    main()