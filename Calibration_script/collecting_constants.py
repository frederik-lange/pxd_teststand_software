"""
 This program collects the constants of several calibrations and collects them in two .csv files, one for values and one for errors.
 The files are called 'output.csv' and 'output_err.csv' and are created in the Calibration_script folder
"""

import configparser
import csv
import os
import pandas as pd

ps = 87
config = configparser.ConfigParser()
config_ini = configparser.ConfigParser()

#config_ini.read("/Users/resi/Desktop/Schreibtisch - MacBook Pro von Theresa/SS2022/BA scibo/Calibrations/ps87/1_Calibration_ps87/constants.ini")

def gain_offset(name_gain, name_offset,channel):
    gain = float(config_ini[f'{channel}'].get(name_gain))
    offset = float(config_ini[f'{channel}'].get(name_offset))
    return gain,offset

def collect_constants():
    path = f"../data/ps{ps}/"
    print(path)
    with open('constants_collection.csv', 'w+', encoding='UTF8') as csvfile:
            names = ["'Date'"] #for constants
            for i in range(24):
                names.append(i)
                names.append("'DAC_VOLTAGE_GAIN_%i'" % i)
                names.append("'DAC_VOLTAGE_OFFSET_%i'" % i)
                names.append("'ADC_U_LOAD_GAIN_%i'" % i)
                names.append("'ADC_U_LOAD_OFFSET_%i'" % i)
                names.append("'ADC_U_REGULATOR_GAIN_%i'" % i)
                names.append("'ADC_U_REGULATOR_OFFSET_%i'" % i)
                names.append("'ADC_I_MON_GAIN_%i'" % i)
                names.append("'ADC_I_MON_OFFSET_%i'" % i)
                names.append("'DAC_CURRENT_GAIN_%i'" % i)
                names.append("'DAC_CURRENT_OFFSET_%i'" % i)

            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()

            for i in range(1, 21):

                    result = os.path.join(path, f"%d_Calibration_ps{ps}" % i + "/constants.ini") #for constants
                    config_ini.read(result)

                    date = str(config_ini[f'Information'].get(f'date')) #for constants
                    values = [date] #for constants
                    for channel in range(24):
                        (g1, o1) = gain_offset(f'DAC_VOLTAGE_GAIN', f'DAC_VOLTAGE_OFFSET', channel)
                        (g2, o2) = gain_offset(f'ADC_U_LOAD_GAIN', f'ADC_U_LOAD_OFFSET', channel)
                        (g3, o3) = gain_offset(f'ADC_U_REGULATOR_GAIN', f'ADC_U_REGULATOR_OFFSET', channel)
                        (g4, o4) = gain_offset(f'ADC_I_MON_GAIN', f'ADC_I_MON_OFFSET', channel)
                        (g5, o5) = gain_offset(f'DAC_CURRENT_GAIN', f'DAC_CURRENT_OFFSET', channel)
                        for i in [channel, g1,o1,g2,o2,g3,o3,g4,o4,g5,o5]:
                            values.append(i)

                    dictonary ={names[i]: values[i] for i in range(len(names))}
                    writer.writerow(dictonary)

    csvfile.close()
    pd.read_csv('constants_collection.csv', header=None).T.to_csv(path + 'output.csv', header=False, index=False)

def collect_errors():
    path = f"../data/ps{ps}/"
    with open('constants_err_collection.csv', 'w+', encoding='UTF8') as csvfile:
            names = [] #for err
            for i in range(24):
                names.append(i)
                names.append("'DAC_VOLTAGE_GAIN_%i'" % i)
                names.append("'DAC_VOLTAGE_OFFSET_%i'" % i)
                names.append("'ADC_U_LOAD_GAIN_%i'" % i)
                names.append("'ADC_U_LOAD_OFFSET_%i'" % i)
                names.append("'ADC_U_REGULATOR_GAIN_%i'" % i)
                names.append("'ADC_U_REGULATOR_OFFSET_%i'" % i)
                names.append("'ADC_I_MON_GAIN_%i'" % i)
                names.append("'ADC_I_MON_OFFSET_%i'" % i)
                names.append("'DAC_CURRENT_GAIN_%i'" % i)
                names.append("'DAC_CURRENT_OFFSET_%i'" % i)

            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()

            for i in range(1, 21):
                    result = os.path.join(path, f"%d_Calibration_ps{ps}" % i + "/constants_err.ini") #for errors
                    config_ini.read(result)
                    values = [] #for errors
                    for channel in range(24):
                        (g1, o1) = gain_offset(f'DAC_VOLTAGE_GAIN', f'DAC_VOLTAGE_OFFSET', channel)
                        (g2, o2) = gain_offset(f'ADC_U_LOAD_GAIN', f'ADC_U_LOAD_OFFSET', channel)
                        (g3, o3) = gain_offset(f'ADC_U_REGULATOR_GAIN', f'ADC_U_REGULATOR_OFFSET', channel)
                        (g4, o4) = gain_offset(f'ADC_I_MON_GAIN', f'ADC_I_MON_OFFSET', channel)
                        (g5, o5) = gain_offset(f'DAC_CURRENT_GAIN', f'DAC_CURRENT_OFFSET', channel)
                        for i in [channel, g1,o1,g2,o2,g3,o3,g4,o4,g5,o5]:
                            values.append(i)

                    dictonary ={names[i]: values[i] for i in range(len(names))}
                    writer.writerow(dictonary)

    csvfile.close()
    pd.read_csv('constants_err_collection.csv', header=None).T.to_csv(path + 'output_err.csv', header=False, index=False)


def main():
    collect_constants()
    collect_errors()

if __name__ == '__main__':
    main()