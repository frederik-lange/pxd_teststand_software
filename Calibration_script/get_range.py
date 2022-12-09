"""
    This program takes the calibration constants (of valid calibrations) and adds them to the constants_range.ini file

"""

import configparser

config_ini = configparser.ConfigParser()
config_range = configparser.ConfigParser()
config_range.optionxform = str

config_range.read(f'./constants_range.ini')

def get_range(name_gain, name_offset,channel):

    gain_upper = float(config_range[f'{channel}'].get(name_gain + '_UPPER'))
    gain_lower = float(config_range[f'{channel}'].get(name_gain + '_LOWER'))
    gain = float(config_ini[f'{channel}'].get(name_gain))

    offset_upper = float(config_range[f'{channel}'].get(name_offset + '_UPPER'))
    offset_lower = float(config_range[f'{channel}'].get(name_offset + '_LOWER'))
    offset = float(config_ini[f'{channel}'].get(name_offset))

    if gain > gain_upper:
        gain_upper = gain
    elif gain < gain_lower:
        gain_lower = gain
    else:
        pass

    if offset > offset_upper:
        offset_upper = offset
    elif offset < offset_lower:
        offset_lower = offset
    else:
        pass

    #config_new[f'{channel}'] = {name_gain + '_UPPER': f'{gain_upper}',
    #                            name_gain + '_LOWER': f'{gain_lower}',
    #                            name_offset + '_UPPER': f'{offset_upper}',
    #                            name_offset + '_LOWER': f'{offset_lower}'}


    return(gain_upper,gain_lower,offset_upper,offset_lower)

def constants_range():
    # Gets the range of constants from 1 .ini-file
    for channel in range(24):
        (g_u_1,g_l_1, o_u_1, o_l_1) = get_range(f'DAC_VOLTAGE_GAIN', f'DAC_VOLTAGE_OFFSET', channel)
        (g_u_2,g_l_2, o_u_2, o_l_2) = get_range(f'ADC_U_LOAD_GAIN', f'ADC_U_LOAD_OFFSET', channel)
        (g_u_3,g_l_3, o_u_3, o_l_3) = get_range(f'ADC_U_REGULATOR_GAIN', f'ADC_U_REGULATOR_OFFSET', channel)
        (g_u_4,g_l_4, o_u_4, o_l_4) = get_range(f'ADC_I_MON_GAIN', f'ADC_I_MON_OFFSET', channel)
        (g_u_5,g_l_5, o_u_5, o_l_5) = get_range(f'DAC_CURRENT_GAIN', f'DAC_CURRENT_OFFSET', channel)

        config_range[f'{channel}'] = {'DAC_VOLTAGE_GAIN_UPPER': g_u_1,
                                    'DAC_VOLTAGE_GAIN_LOWER': g_l_1,
                                    'DAC_VOLTAGE_OFFSET_UPPER': o_u_1,
                                    'DAC_VOLTAGE_OFFSET_LOWER': o_l_1,
                                    'ADC_U_LOAD_GAIN_UPPER': g_u_2,
                                    'ADC_U_LOAD_GAIN_LOWER': g_l_2,
                                    'ADC_U_LOAD_OFFSET_UPPER': o_u_2,
                                    'ADC_U_LOAD_OFFSET_LOWER': o_l_2,
                                    'ADC_U_REGULATOR_GAIN_UPPER': g_u_3,
                                    'ADC_U_REGULATOR_GAIN_LOWER': g_l_3,
                                    'ADC_U_REGULATOR_OFFSET_UPPER': o_u_3,
                                    'ADC_U_REGULATOR_OFFSET_LOWER': o_l_3,
                                    'ADC_I_MON_GAIN_UPPER': g_u_4,
                                    'ADC_I_MON_GAIN_LOWER': g_l_4,
                                    'ADC_I_MON_OFFSET_UPPER': o_u_4,
                                    'ADC_I_MON_OFFSET_LOWER': o_l_4,
                                    'DAC_CURRENT_GAIN_UPPER': g_u_5,
                                    'DAC_CURRENT_GAIN_LOWER': g_l_5,
                                    'DAC_CURRENT_OFFSET_UPPER': o_u_5,
                                    'DAC_CURRENT_OFFSET_LOWER': o_l_5}

        with open('constants_range.ini', 'w+') as configfile:
            config_range.write(configfile)

def main():
    # Get the range of constants for several calibrations
    ps = 15
    for cal in range(1,21):
        config_ini.read(f'../data/ps{ps}/{cal}_Calibration_ps{ps}/constants.ini')
        constants_range()
        print(f"Constants of calibration {cal} collected.")

if __name__ == '__main__':
    main()
