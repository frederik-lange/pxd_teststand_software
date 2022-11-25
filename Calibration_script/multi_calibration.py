import configparser
import numpy as np

from Calibration_script import main
import os

ps = 87
config = configparser.ConfigParser()
config.read('path.ini')
config['calibration_data']['multi_cal_path'] = f'../data/ps{ps}'

#print
for val in config['calibration_data']:
    print(val,config['calibration_data'][val])

bad_data = np.zeros(20)

# loop
for i in range(10,12):
    print(f"\nPS {ps} Calibration {i} in progress...\n")
    config['calibration_data']['data_path'] = config['calibration_data'].get('multi_cal_path') + f'/{i}_Calibration_ps{ps}'
    with open('path.ini', 'w') as configfile:
        config.write(configfile)
    try:
        main.main()
        print(f"\nPS {ps} Calibration {i} done!\n")
    except (TypeError):
        bad_data[i-1] = 1
        print(f"PS {ps} Calibration {i} raised an error!")


"""
if __name__ == '__main__':
    main()
"""