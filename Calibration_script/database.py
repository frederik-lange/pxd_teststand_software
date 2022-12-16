"""
    Collects data from ALL the constants.ini files in the CalibrationData folder and stores them in a .csv file
    Requires a working folder structure!
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

path = '../data/CalibrationData/'
#ps = 'unknown'

names = ["Unit","Date"]
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
    with open('../data/database.csv', 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        writer.writeheader()

# read all constants.ini files:
def scan_and_add(path, ps):
    with open('../data/database.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        n = 0
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
    values = [ps,date]
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
    # dont add values that have 0 or 1e8
    add = True
    for i in dict:
        #print(i, dict[i])
        if dict[i] == '0' or dict[i] == '100000000':
            print('Invalid values!')
            add = False
    if add == True:
        writer.writerow(dict)

def fill():
    initialize()
    # go through all possible ps folders
    for i in range(21,100):
        path_ps = os.path.join(path,f"ps{i}")
        if os.path.exists(path_ps):
            ps = i
            print(f"PS number {i}")
            scan_and_add(path_ps,ps)

def update_range():
    data = pd.read_csv('../data/database.csv')
    print(data['Unit'].shape)
    print(data.shape)
    medians = np.zeros(240)
    sigmas = np.zeros(240)
    print(medians.shape)

    print('Median',names[2],np.mean(data[names[2]]))
    medians[:] = np.mean(data[names[2:]])
    medians = medians.reshape(24,10)
    print('STD',names[2],np.std(data[names[2]]))
    sigmas[:] = np.std(data[names[2:]])
    sigmas = sigmas.reshape(24,10)
    #print(medians[:][0])
    #print(medians)

    config = configparser.ConfigParser()

    for channel in range(24):
        config[f'{channel}'] = {
            names[2 + channel*10 + n] : medians[channel][n] for n in range(10)
        }
    with open('../data/database.ini', 'w') as configfile:
        config.write(configfile)
    for channel in range(24):
        config[f'{channel}'] = {
            names[2 + channel*10 + n] : sigmas[channel][n] for n in range(10)
        }
    with open('../data/database_std.ini', 'w') as configfile:
        config.write(configfile)

def gauss(x, mu, sigma, a):
    return a * 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/2/sigma**2)

def normal_distribution():
    data = pd.read_csv('./../data/database.csv')
    config_vals = configparser.ConfigParser()
    config_vals.read('../data/database.ini')
    config_errs = configparser.ConfigParser()
    config_errs.read('../data/database_std.ini')
    print(data.shape)
    with PdfPages('../data/Normal_Distribution') as pdf:
        for channel in range(24):
            print(channel)
            for n in range(10):
                plt.subplots()
                plt.xlabel('Values')
                plt.ylabel('Counts')
                plt.title(f'Channel {channel}: {names[2 + channel * 10 + n]}')
                print(names[2+channel*10+n])
                med, std = float(config_vals[f'{channel}'][names[2+channel*10+n]]), float(config_errs[f'{channel}'][names[2+channel*10+n]])
                max = int(np.max(data[names[2+channel*10+n]]))
                min = int(np.min(data[names[2+channel*10+n]]))
                diff = max-min
                if diff > 200:
                    diff = 200
                histo = plt.hist(data[names[2+channel*10+n]],bins=diff)
                #print(histo[0], "\n", histo[1])
                plt.axvline(med, color='black')
                plt.axvline(med - std, color='green')
                plt.axvline(med + std, color='green')
                plt.axvline(med - 2*std, color='yellow')
                plt.axvline(med + 2*std, color='yellow')
                plt.axvline(med + 3*std, color='red')
                plt.axvline(med - 3*std,color='red')
                x = np.arange(min,max,0.1)
                try:
                    popt, pcov = so.curve_fit(gauss, histo[1][1:] - histo[1][:-1], histo[0], bounds=([med-std,0,0],[med+std,2*std,2*len(data[names[2+channel*10+n]])]))
                    print(popt,pcov)
                    plt.plot(x, gauss(x,*popt))
                except RuntimeError:
                    pass
                pdf.savefig()
                plt.close()

def main():
    #fill()
    update_range()
    #normal_distribution()
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