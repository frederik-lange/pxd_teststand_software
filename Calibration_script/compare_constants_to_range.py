"""
    Compares the calibration constants of a new calibration to the existing range and shows the results visually
    Results are found in the folder with the source files in 'compare_constants_to_range.pdf'
    Note: Irrelevant!
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from matplotlib.backends.backend_pdf import PdfPages

path = '../data/MainProdNode47/'
channels = np.arange(0, 24, 1)
config_range = configparser.ConfigParser()
config_range.read('./constants_range.ini')
config = configparser.ConfigParser()
config.read(os.path.join(path, 'constants.ini'))
config_err = configparser.ConfigParser()
config_err.read(os.path.join(path, 'constants_err.ini'))
gains, offsets = np.zeros(24), np.zeros(24)
gain_errs, offset_errs = np.zeros(24), np.zeros(24)
upper_gains, lower_gains = np.zeros(24), np.zeros(24)
upper_offsets, lower_offsets = np.zeros(24), np.zeros(24)

def plot_gains(pdf, channel, values, errors, range_upper, range_lower,title):
    plt.subplots()
    plt.grid()
    plt.xlim(-1,24)
    plt.xticks(channels)
    plt.xlabel('Channel')
    plt.ylabel('Gains normalized to lower range')
    norm_values = np.zeros_like(values)
    for v in range(len(values)):
        if range_upper[v] > 0:
            norm_values[v] = range_upper[v]/range_lower[v]
        else:
            norm_values[v] = np.abs(range_lower[v]/range_upper[v])
    for i in channels:
        if i == 0:
            plt.bar(i, bottom=1, height=norm_values[i]-1, alpha=0.5, color='g',label='previous range')
        else:
            plt.bar(i, bottom=1, height=norm_values[i]-1, alpha=0.5, color='g')
    inrange = np.zeros_like(values,dtype=bool)
    deviation = np.zeros_like(values)
    for i in range(len(norm_values)):
        if (values[i] < range_upper[i] and values[i] > range_lower[i]):
            inrange[i] = True
        elif values[i] > range_upper[i]:
            deviation[i] = np.abs((values[i] - range_upper[i]) / range_upper[i])
        elif values[i] < range_lower[i]:
            deviation[i] = np.abs((values[i] - range_lower[i]) / range_lower[i])
    for d in deviation:
        if d!=0:
            print(f'Deviation: {d:.2%}')
    #print(inrange)
    plt.scatter(channels[inrange], norm_values[inrange],color='green', label='new constants in range')
    plt.scatter(channels[~inrange], norm_values[~inrange], color='red', label='new constants out of range')
    #plt.errorbar(channels,norm_values,yerr=errors/range_lower,fmt='ob')
    plt.title(title)
    plt.legend()
    pdf.savefig()
    plt.close()
    return None

def plot_offsets(pdf,channel,values,errors,range_upper,range_lower,title):
    plt.subplots()
    plt.grid()
    plt.xlim(-1, 24)
    plt.xticks(channels)
    plt.xlabel('Channel')
    plt.ylabel('Offsets')
    for i in channels:
        if i == 0:
            plt.bar(i, bottom=range_lower[i], height=range_upper[i]-range_lower[i], alpha=0.5, color='g',label='previous range')
        else:
            plt.bar(i, bottom=range_lower[i], height=range_upper[i] - range_lower[i], alpha=0.5, color='g')
    inrange = np.zeros_like(values, dtype=bool)
    deviation = np.zeros_like(values)
    for i in range(len(values)):
        if (values[i] < range_upper[i] and values[i] > range_lower[i]):
            inrange[i] = True
        elif values[i] > range_upper[i]:
            deviation[i] = np.abs((values[i] - range_upper[i])/range_upper[i])
        elif values[i] < range_lower[i]:
            deviation[i] = np.abs((values[i] - range_lower[i])/range_lower[i])
    for d in deviation:
        if d!=0:
            print(f'Deviation: {d:.2%}')
    plt.scatter(channels[inrange], values[inrange], color='g', label='new constants in range')
    plt.scatter(channels[~inrange], values[~inrange], color='r', label='new constants out of range')
    # plt.errorbar(channels,norm_values,yerr=errors/range_lower,fmt='ob')
    plt.title(title)
    plt.legend()
    pdf.savefig()
    plt.close()
    return None

def get_constants(pdf,name):
    for channel in range(24):
        #print(f"Plotting Channel {channel}...")
        gains[channel] = config[f'{channel}'][name+'_GAIN']
        offsets[channel] = config[f'{channel}'][name+'_OFFSET']
        gain_errs[channel] = config_err[f'{channel}'][name+'_GAIN']
        offset_errs[channel] = config_err[f'{channel}'][name+'_OFFSET']
        upper_gains[channel] = config_range[f'{channel}'][name+'_GAIN_UPPER']
        lower_gains[channel] = config_range[f'{channel}'][name+'_GAIN_LOWER']
        upper_offsets[channel] = config_range[f'{channel}'][name+'_OFFSET_UPPER']
        lower_offsets[channel] = config_range[f'{channel}'][name+'_OFFSET_LOWER']
        #print(gains[channel],upper_gains[channel],lower_gains[channel])
    #print(gains)
    #print(upper_gains)
    #print(lower_gains)
    plot_gains(pdf, channel, gains, gain_errs, upper_gains, lower_gains, name + f' Gain (normalized to lower range constant)')
    plot_offsets(pdf, channel, offsets, offset_errs, upper_offsets, lower_offsets, name + f' offset')
    return gains,offsets,gain_errs,offset_errs,upper_gains,lower_gains,upper_offsets,lower_offsets

def main():
    print(config['Information']['date'])
    with PdfPages(os.path.join(path, 'compare_constants_to_range.pdf')) as pdf:
        get_constants(pdf,'DAC_VOLTAGE')
        get_constants(pdf,'ADC_U_LOAD')
        get_constants(pdf,'ADC_U_REGULATOR')
        get_constants(pdf,'ADC_I_MON')
        get_constants(pdf,'DAC_CURRENT')
    #for i in range(len(upper_gains)):
    #        print(upper_gains[i], lower_gains[i])

# NOTE: RANGE IS WRONG WHEN UPPER AND LOWER END CROSS ZERO!


"""
    with PdfPages(os.path.join(path,'compare_constants_to_range.pdf')) as pdf:

        for channel in range (24):

            # new values:
            gain1, offset1 = config[f'{channel}']['DAC_VOLTAGE_GAIN'], config[f'{channel}']['DAC_VOLTAGE_OFFSET']
            gain2, offset2 = config[f'{channel}']['ADC_U_LOAD_GAIN'], config[f'{channel}']['ADC_U_LOAD_OFFSET']
            gain3, offset3 = config[f'{channel}']['ADC_U_REGULATOR_GAIN'], config[f'{channel}']['ADC_U_REGULATOR_OFFSET']
            gain4, offset4 = config[f'{channel}']['ADC_I_MON_GAIN'], config[f'{channel}']['ADC_I_MON_OFFSET']
            gain5, offset5 = config[f'{channel}']['DAC_CURRENT_GAIN'], config[f'{channel}']['DAC_CURRENT_OFFSET']
            # range values: g = gain, o = offset, u = upper, l = lower
            gu1, gl1, ou1, ol1 = config_range[f'{channel}']['DAC_VOLTAGE_GAIN_UPPER'], config_range[f'{channel}']['DAC_VOLTAGE_GAIN_LOWER'], config_range[f'{channel}']['DAC_VOLTAGE_OFFSET_UPPER'], config_range[f'{channel}']['DAC_VOLTAGE_OFFSET_LOWER']
            gu2, gl2, ou2, ol2 = config_range[f'{channel}']['ADC_U_LOAD_GAIN_UPPER'], config_range[f'{channel}']['ADC_U_LOAD_GAIN_LOWER'], config_range[f'{channel}']['ADC_U_LOAD_OFFSET_UPPER'], config_range[f'{channel}']['ADC_U_LOAD_OFFSET_LOWER']
            gu3, gl3, ou3, ol3 = config_range[f'{channel}']['ADC_U_REGULATOR_GAIN_UPPER'], config_range[f'{channel}']['ADC_U_REGULATOR_GAIN_LOWER'], config_range[f'{channel}']['ADC_U_REGULATOR_OFFSET_UPPER'], config_range[f'{channel}']['ADC_U_REGULATOR_OFFSET_LOWER']
            gu4, gl4, ou4, ol4 = config_range[f'{channel}']['ADC_I_MON_GAIN_UPPER'], config_range[f'{channel}']['ADC_I_MON_GAIN_LOWER'], config_range[f'{channel}']['ADC_I_MON_OFFSET_UPPER'], config_range[f'{channel}']['ADC_I_MON_OFFSET_LOWER']
            gu5, gl5, ou5, ol5 = config_range[f'{channel}']['DAC_CURRENT_GAIN_UPPER'], config_range[f'{channel}']['DAC_CURRENT_GAIN_LOWER'], config_range[f'{channel}']['DAC_CURRENT_OFFSET_UPPER'], config_range[f'{channel}']['DAC_CURRENT_OFFSET_LOWER']
"""

if __name__ == '__main__':
    main()