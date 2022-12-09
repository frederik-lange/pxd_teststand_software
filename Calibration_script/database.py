"""
    Collects data from the constants.ini files and stores them in a .csv file
"""

import numpy as np
import csv
import configparser
import os

path = '../data'
ps = 'unknown'

names = ["'Unit'","'Date'"]
for i in range(24):
    names.append(f"DAC_VOLTAGE_GAIN_{i}")
    names.append(f"DAC_VOLTAGE_OFFSET_{i}")
    names.append(f"ADC_U_LOAD_GAIN_{i}")
    names.append(f"ADC_U_LOAD_OFFSET_{i}")
    names.append(f"ADC_U_REGULATOR_GAIN_{i}")
    names.append(f"ADC_U_REGULATOR_OFFSET{i}")
    names.append(f"ADC_I_MON_GAIN_{i}")
    names.append(f"ADC_I_MON_OFFSET_{i}")
    names.append(f"DAC_CURRENT_GAIN_{i}")
    names.append(f"DAC_CURRENT_OFFSET_{i}")

# initialize database:
# read all constants.ini files:
def initialize(writer):
    n = 0
    for root, dirs, files in os.walk('../data'):
        for file in files:
            if file.endswith('constants.ini'):
                path_file = os.path.join(root,file)
                n += 1
                try:
                    add_constants(writer, path_file)
                except KeyError:
                    print(f'File {path_file} did not work!')
    print(f"{n} Files have been scanned")

def add_constants(writer, file):
    config = configparser.ConfigParser()
    config.read(file)
    date = str(config['Information']['date'])
    print(date)
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
    print(dict)
    writer.writerow(dict)

def main():
    file = '../data/constants.ini'
    with open('../data/database.csv', 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        writer.writeheader()
        initialize(writer)
        #add_constants(writer)


if __name__ == '__main__':
    main()