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
    config = configparser.ConfigParser()
    config.read('path.ini')
    for i in range(20,90):
        for root, dirs, files in os.walk(f'../data/CalibrationData/ps{i}'):
            for dir in dirs:
                if os.path.exists(os.path.join(root,dir,'Channel_0_U_vs_U.dat')):
                    print(f'{root}/{dir}')
                    config['calibration_data']['data_path'] = os.path.join(root,dir)
                    with open('path.ini', 'w') as configfile:
                        config.write(configfile)
                    #Calibration_script.main.main()
                    try:
                        Calibration_script.main.main()
                        print(f"\nPS {i} Calibration done!\n")
                    except (TypeError):
                        print(f"PS {i} Calibration raised an error!")
    '''
    ps = 26
    config = configparser.ConfigParser()
    config.read('path.ini')
    config['calibration_data']['multi_cal_path'] = f'../data/ps{ps}_monitoring'

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
    '''


if __name__ == '__main__':
    main()
