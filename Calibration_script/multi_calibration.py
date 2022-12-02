"""
Can run the main.py script multiple times in several folders
Takes the path form multi_cal_path and then sets data_path
"""

import configparser
import numpy as np

import Calibration_script.main
from Calibration_script import main
import os

def main():
    ps = 87
    config = configparser.ConfigParser()
    config.read('path.ini')
    config['calibration_data']['multi_cal_path'] = f'../data/ps{ps}'

    for val in config['calibration_data']:
        print(val,config['calibration_data'][val])

    bad_data = np.zeros(20)

    # loop
    for i in range(1,21):
        print(f"\nPS {ps} Calibration {i} in progress...\n")
        config['calibration_data']['data_path'] = config['calibration_data'].get('multi_cal_path') + f'/{i}_Calibration_ps{ps}'
        with open('path.ini', 'w') as configfile:
            config.write(configfile)
        try:
            Calibration_script.main.main()
            print(f"\nPS {ps} Calibration {i} done!\n")
        except (TypeError):
            bad_data[i-1] = 1
            print(f"PS {ps} Calibration {i} raised an error!")

if __name__ == '__main__':
    main()
