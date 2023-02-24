import os
import configparser
from Calibration_script import main
import numpy as np

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

check_all()