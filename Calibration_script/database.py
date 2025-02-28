"""
    Collects data from ALL the constants.ini files in the CalibrationData folder and stores them in a .csv file
    Requires a working folder structure!
    The pdf is stored in "Normal_Distribution.pdf" in the data folder
"""

import numpy as np
import csv
import configparser
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import scipy.optimize as so

#database = 'database'
database = 'PS_105_constants'
path = '../data/CalibrationData/ps105'
#ps = 'unknown'

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

# initialize database:
def initialize():
    '''
    Warning! This function deletes all data and creates a new database!
    :return: None
    '''
    with open(f'../data/{database}.csv', 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        writer.writeheader()

# read all constants.ini files:
def scan_and_add(path, ps):
    with open(f'../data/{database}.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        n, invalid = 0, 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('constants.ini'):
                    path_file = os.path.join(root,file)
                    n += 1
                    try:
                        add_constants(writer, path_file, ps)
                    except KeyError:
                        print(f'File {path_file} did not work!')
    print(f"{n} Files have been scanned")

def add_constants(writer, file, ps):
    config = configparser.ConfigParser()
    config.read(file)
    date = str(config['Information']['date'])
    #print(date)
    autom_success = config['Information']['success']
    if autom_success == 'True':
        validated = 'yes'
    else: validated = 'no'
    values = [ps,date,validated,'no']
    for channel in range(24):
        values.append(config[f'{channel}']['DAC_VOLTAGE_GAIN'])
        values.append(config[f'{channel}']['DAC_VOLTAGE_OFFSET'])
        values.append(config[f'{channel}']['ADC_U_LOAD_GAIN'])
        values.append(config[f'{channel}']['ADC_U_LOAD_OFFSET'])
        values.append(config[f'{channel}']['ADC_U_REGULATOR_GAIN'])
        values.append(config[f'{channel}']['ADC_U_REGULATOR_OFFSET'])
        values.append(config[f'{channel}']['ADC_I_MON_GAIN'])
        values.append(config[f'{channel}']['ADC_I_MON_OFFSET'])
        values.append(config[f'{channel}']['DAC_CURRENT_GAIN'])
        values.append(config[f'{channel}']['DAC_CURRENT_OFFSET'])
    dict = {names[i] : values[i] for i in range(len(names))}
    # check if values are good
    # dont add values that have 0 or 1e8 (these are default constants)
    #print(config['Information'].get('success'))
    add = True
    for i in dict:
        #print(i, dict[i])
        if dict[i] == '0' or dict[i] == '100000000':
            add = False
    if add == True:
        writer.writerow(dict)
    else:
        print('Invalid values!')

def fill():
    initialize()
    # go through all possible ps folders
    for i in range(21,110):
        path_ps = os.path.join(path,f"ps{i}")
        if os.path.exists(path_ps):
            ps = i
            print(f"PS number {i}")
            scan_and_add(path_ps,ps)

def update_range():
    # updates the range of valid constants. Results can be found in the 'database.ini' and 'database_std.ini'
    data = pd.read_csv(f'../data/{database}.csv')
    #print(data['Unit'].shape)
    #print(data.shape)
    medians = np.zeros(240)
    sigmas = np.zeros(240)

    mask = data['used_for_range'] == 'yes'
    #mask = np.ones(len(data[names[0]]),dtype=bool)
    #print('Median',names[2],np.mean(data[names[2]]))
    medians[:] = np.mean(data[names[4:]][mask])
    medians = medians.reshape(24,10)
    #print('STD',names[2],np.std(data[names[2]]))
    sigmas[:] = np.std(data[names[4:]][mask])
    sigmas = sigmas.reshape(24,10)
    #print(medians[:][0])
    #print(medians)

    # special case for hv (Channel 13): consider only values with muA monitoring
    # not needed when considering used_for_range
    mask = data['ADC_I_MON_GAIN_13']>-1500
    medians[13,6] = np.mean(data['ADC_I_MON_GAIN_13'][mask])
    sigmas[13,6] = np.std(data['ADC_I_MON_GAIN_13'][mask])
    medians[13,7] = np.mean(data['ADC_I_MON_OFFSET_13'][mask])
    sigmas[13,7] = np.std(data['ADC_I_MON_OFFSET_13'][mask])
    mask = data['DAC_CURRENT_GAIN_13']>90000
    medians[13,8] = np.mean(data['DAC_CURRENT_GAIN_13'][mask])
    sigmas[13,8] = np.std(data['DAC_CURRENT_GAIN_13'][mask])
    medians[13,9] = np.mean(data['DAC_CURRENT_OFFSET_13'][mask])
    sigmas[13,9] = np.std(data['DAC_CURRENT_OFFSET_13'][mask])
    # special case for bulk (Channel 15): consider only values with muA monitoring
    mask = data['ADC_I_MON_GAIN_15'] > -500000
    medians[15,6] = np.mean(data['ADC_I_MON_GAIN_15'][mask])
    sigmas[15,6] = np.std((data['ADC_I_MON_GAIN_15'][mask]))
    medians[15,7] = np.mean(data['ADC_I_MON_OFFSET_15'][mask])
    sigmas[15,7] = np.std((data['ADC_I_MON_OFFSET_15'][mask]))
    #print(medians[15,6:8],sigmas[15,6:8])

    config = configparser.ConfigParser()

    # cut away name endings for some .ini-files:
    if False:
        for n in range(4,len(names)):
            if n<104:
                names[n] = str(names[n])[:-2]
            else:
                names[n] = str(names[n])[:-3]

    for channel in range(24):
        config[f'{channel}'] = {
            str(names[4 + channel*10 + n]) : medians[channel][n] for n in range(10)
        }
    with open(f'../data/{database}.ini', 'w') as configfile:
        config.write(configfile)
    for channel in range(24):
        config[f'{channel}'] = {
            str(names[4 + channel*10 + n]) : sigmas[channel][n] for n in range(10)
        }
    with open(f'../data/{database}_std.ini', 'w') as configfile:
        config.write(configfile)
    print("Range of calibration constants was updated!")

def ratio_mean_std():
    config_vals = configparser.ConfigParser()
    config_vals.read(f'../data/{database}.ini')
    config_errs = configparser.ConfigParser()
    config_errs.read(f'../data/{database}_std.ini')
    plotnames=['DAC_VOLTAGE_GAIN','DAC_VOLTAGE_OFFSET','ADC_U_LOAD_GAIN','ADC_U_LOAD_OFFSET','ADC_U_REGULATOR_GAIN','ADC_U_REGULATOR_OFFSET',
           'ADC_I_MON_GAIN','ADC_I_MON_OFFSET','DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/Constants_variance_{database}.pdf') as pdf:
        for n in range(10):
            x = np.arange(0,24,1)
            y = np.zeros(24)
            for channel in range(24):
                med, std = float(config_vals[f'{channel}'][names[4 + channel * 10 + n]]), float(config_errs[f'{channel}'][names[4 + channel * 10 + n]])
                varK = np.abs(std/med)
                y[channel] = varK
            plt.figure()
            plt.grid()
            plt.xlabel('Channels'),plt.title(plotnames[n])
            plt.ylabel('Constants Variance')
            plt.xticks(np.arange(0,24,1))
            plt.bar(x,y)
            pdf.savefig()

def gauss(x, mu, sigma, a):
    return a * 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/2/sigma**2)

def normal_distribution():
    data = pd.read_csv(f'./../data/{database}.csv')
    config_vals = configparser.ConfigParser()
    config_vals.read(f'../data/{database}.ini')
    config_errs = configparser.ConfigParser()
    config_errs.read(f'../data/{database}_std.ini')
    #print(data.shape)
    with PdfPages(f'../data/Normal_Distribution_{database}') as pdf:
        count_1, count_tot_1, count_2, count_tot_2, count_3, count_tot_3, count_4, count_tot_4 = 0, 0, 0, 0, 0, 0, 0, 0
        for channel in range(24):
            print(f'Plotting channel {channel}...')
            for n in range(10):
                plt.subplots()
                plt.xlabel('Values')
                plt.ylabel('Counts')
                plt.title(f'Channel {channel}: {names[4 + channel * 10 + n]}')
                #print(names[2+channel*10+n])
                if False:
                    for x in range(4, len(names)):
                        if x < 104:
                            names[n] = str(names[n])[:-2]
                        else:
                            names[n] = str(names[n])[:-3]
                med, std = float(config_vals[f'{channel}'][names[4+channel*10+n]]), float(config_errs[f'{channel}'][names[4+channel*10+n]])
                #mask = np.full(len(data[names[2+channel*10+n]]), True)
                if channel == 13 and (n == 6 or n == 7):
                    mask = (data['ADC_I_MON_GAIN_13'] > -1500) & (data['used_for_range'] == 'yes')
                elif channel == 13 and (n==8 or n == 9):
                    mask = (data['DAC_CURRENT_GAIN_13'] > 90000) & (data['used_for_range'] == 'yes')
                elif channel == 15 and (n==6 or n == 7):
                    mask = (data['ADC_I_MON_GAIN_15'] > -500000) & (data['used_for_range'] == 'yes')
                else:
                    mask = data['used_for_range'] == 'yes'
                max = int(np.max(data[names[4+channel*10+n]][mask]))
                min = int(np.min(data[names[4+channel*10+n]][mask]))
                diff = max-min
                if diff > 100:
                    diff = 100
                if diff < 1:
                    diff = 1
                histo = plt.hist(data[names[4+channel*10+n]][mask],bins=diff)
                #print(histo[0], "\n", histo[1])
                plt.axvline(med, color='black', label='mean')
                plt.axvline(med - std, color='green', label='$1 \sigma$')
                plt.axvline(med + std, color='green')
                plt.axvline(med - 2*std, color='yellow', label='$2 \sigma$')
                plt.axvline(med + 2*std, color='yellow',)
                plt.axvline(med + 3*std, color='blue', label='$3 \sigma$')
                plt.axvline(med - 3*std,color='blue')
                plt.axvline(med + 4*std, color='red', label='$4 \sigma$')
                plt.axvline(med - 4*std, color='red')
                x = np.arange(min,max,0.1)
                try:
                    popt, pcov = so.curve_fit(gauss, (histo[1][1:]+histo[1][:-1])/2, histo[0], bounds=([med-1*std,0,0],[med+1*std,1*std,1*len(data[names[2+channel*10+n]])]))
                    #print(popt)
                    x = np.arange(med-3*std, med+3*std,0.1)
                    plt.plot(x, gauss(x,*popt),label='fitted gauss curve')
                except (RuntimeError, ValueError):
                    print("Gauss function could not be fitted!")

                plt.legend()
                # check if values are normally distributed
                if n%2==0:
                    for i in range(1,5):
                        mask = (histo[1][:-1] > med - i*std) & (histo[1][:-1] < med+i*std)
                        sum, sum_total = np.sum(histo[0][mask]), np.sum(histo[0])
                        #print(sum, sum_total, sum/sum_total)
                        if i == 1:
                            count_tot_1 += 1
                            if sum/sum_total >=  0.6827:
                                count_1 += 1
                        if i == 2:
                            count_tot_2 += 1
                            if sum/sum_total >= 0.9545:
                                count_2 += 1
                        if i == 3:
                            count_tot_3 += 1
                            if sum/sum_total >= 0.9973:
                                count_3 += 1
                        if i == 4:
                            count_tot_4 += 1
                            if sum/sum_total >= 0.99993666:
                                count_4 += 1
                pdf.savefig()
                plt.close()
        print(f'Normally distributed values: \n1 sigma: {count_1} of {count_tot_1}\n2 sigma: {count_2} of {count_tot_2}\n3 sigma: {count_3} of {count_tot_3}\n4 sigma: {count_4} of {count_tot_4}')

def main():
    # warning! unse fill() only for full database!
    #fill()
    #scan_and_add(path,105)
    #define_range()
    #update_range()
    #normal_distribution()
    ratio_mean_std()
    '''
    source = '/home/silab44/pxd_teststand_software_git/pxd_teststand_software/OldCallibrations'
    dest = '/home/silab44/pxd_teststand_software_frederik/data/CalibrationData'
    print(os.path.expanduser('~'))
    for i in range(32,90):
        try:
            shutil.copytree(os.path.join(source,f'MainProdNode{i}'), os.path.join(dest,f'ps{i}/MainProdNode{i}'))
        except:
            ValueError
    '''

if __name__ == '__main__':
    main()