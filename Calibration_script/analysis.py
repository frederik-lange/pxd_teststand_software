import os
import configparser
from Calibration_script import main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv

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

vars = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET', 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']

channel_names = ['(dhp-io)','(sw-dvdd)','(dcd-dvdd)','(dhp-core)','(dcd-refin)','(source)','(dcd-avdd)','(amplow)','(ccg1)','(ccg2)','(drift)','(ccg3)','(poly)',
                 '(HV)','(guard)','(bulk)','(gate-on1)','(gate-on2)','(gate-off)','(gate-on3)','(clear-on)','(sw-refin)','(sw-sub)','(clear-off)']
group1 = [0, 1, 2, 3, 4, 6, 7]
group2 = [8, 9, 10, 11]
group3 = [16, 17, 18, 19]
group4 = [20, 23]
group5 = [21, 22]
group6 = [5, 12, 13, 14, 15]

def check_all():
    # compare all calibration constants to the valid range
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

def boxplot_per_channel():
    # Plot boxplots for 10 constants per Channel
    data = pd.read_csv(f'./../data/database.csv')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/database_boxplots_per_channel.pdf') as pdf:
        for channel in range(24):
            print(f"Plotting Channel {channel}...")
            y = np.zeros((10,len(data[names[0]])))
            list = []
            for n in range(10):
                y[n] = data[names[4 + n*24 + channel]]
                list.append(y[n])
            fig,ax = plt.subplots(2,5)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            fig.suptitle(f'Channel {channel}')
            for n in range(10):
                if n <= 4:
                    a = 0
                else:
                    a = 1
                plt.grid()
                ax[a,n%5].boxplot(y[n])
                ax[a,n%5].set_title(plotnames[n])
            plt.tight_layout()
            plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9)
            pdf.savefig()
            plt.close(fig)

def boxplot_per_constant():
    # Plot boxplots for 24 channels per constant
    data = pd.read_csv(f'./../data/database.csv')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    #with PdfPages(f'../data/database_boxplots_per_constant.pdf') as pdf:
    x = np.arange(0, 24, 1)
    print(x)
    for n in range(10):
        with PdfPages(f'/home/silab44/Desktop/Frederik/Plots/boxplots_{vars[n]}.pdf') as pdf:
                print(f"Plotting {plotnames[n]}...")
                mask = data['used_for_range'] == 'yes'
                fig, ax = plt.subplots()
                fig.set_figheight(4)
                fig.set_figwidth(7)
                y = np.zeros((24, len(data[names[0]][mask])))
                list = []
                for channel in range(24):
                    y[channel] = data[names[4 + channel*10 + n]][mask]
                    list.append(y[channel])
                plt.xlabel('Channels')
                plt.title(plotnames[n].replace("_", " "))
                ax.boxplot(list, positions=x)
                plt.xticks(x)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)

def constant_precision():
    pass

def channel_grouping_by_board():
    # group channels the way they are placed on the boards
    config_mean, config_std = configparser.ConfigParser(), configparser.ConfigParser()
    config_mean.read('../data/database.ini')
    config_std.read('../data/database_std.ini')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/channel_grouping_by_board.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            plt.title(plotnames[n])
            fig, ax = plt.subplots(2,3)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            fig.suptitle(plotnames[n])
            means, stds = np.zeros(24), np.zeros(24)
            for channel in range(24):
                means[channel] = float(config_mean[f'{channel}'][plotnames[n]+f'_{channel}'])
                stds[channel] = float(config_std[f'{channel}'][plotnames[n]+f'_{channel}'])
            for plot in range(6):
                x = np.arange(plot*4,plot*4+4,1)
                xt = x
                for i in xt:
                    i = str(i)
                x = np.arange(0, len(x), 1)
                y_mean = means[plot*4:plot*4+4]
                y_std = stds[plot*4:plot*4+4]
                if plot<3:
                    ax[0,plot%3].bar(x,2*y_std,bottom=y_mean-y_std,alpha=1)
                    ax[0, plot % 3].hlines(y_mean,x-0.5,x+0.5)
                    ax[0, plot % 3].set_xticks(x)
                    ax[0, plot % 3].set_xticklabels(xt)
                else:
                    ax[1, plot % 3].bar(x, 2 * y_std, bottom=y_mean - y_std,alpha=1)
                    ax[1, plot % 3].hlines(y_mean, x - 0.5, x + 0.5)
                    ax[1, plot % 3].set_xticks(x)
                    ax[1, plot % 3].set_xticklabels(xt)
                plt.xticks(x)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

def channel_grouping():
    # Plot constants as bars for channels sorted by manually evaluated groups
    config_mean, config_std = configparser.ConfigParser(), configparser.ConfigParser()
    config_mean.read('../data/database.ini')
    config_std.read('../data/database_std.ini')
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    groups = [group1, group2, group3, group4, group5, group6]
    with PdfPages(f'../data/channel_grouping.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
            plt.title(plotnames[n])
            fig, ax = plt.subplots(2, 3)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            fig.suptitle(plotnames[n].replace("_", " "))
            means, stds = np.zeros(24), np.zeros(24)
            for channel in range(24):
                means[channel] = float(config_mean[f'{channel}'][plotnames[n] + f'_{channel}'])
                stds[channel] = float(config_std[f'{channel}'][plotnames[n] + f'_{channel}'])
            for plot in range(6):
                x = np.array(groups[plot])
                xt = x
                for i in xt:
                    i = str(i)
                x = np.arange(0,len(x),1)
                #y = np.array
                y_mean = means[np.array(groups[plot])]
                y_std = stds[np.array(groups[plot])]
                if plot < 3:
                    ax[0, plot % 3].bar(x, 2 * y_std, bottom=y_mean - y_std, alpha=1)
                    ax[0, plot % 3].hlines(y_mean, x - 0.5, x + 0.5)
                    ax[0, plot % 3].set_xticks(x)
                    ax[0, plot % 3].set_xticklabels(xt)
                else:
                    ax[1, plot % 3].bar(x, 2 * y_std, bottom=y_mean - y_std, alpha=1)
                    ax[1, plot % 3].hlines(y_mean, x - 0.5, x + 0.5)
                    ax[1, plot % 3].set_xticks(x)
                    ax[1, plot % 3].set_xticklabels(xt)
                #plt.xticks(xt)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

def grouping_boxplots():
    # boxplots for all channels grouped
    data = pd.read_csv('./../data/database.csv')
    mask = data['used_for_range'] == 'yes'
    plotnames = ['DAC_VOLTAGE_GAIN', 'DAC_VOLTAGE_OFFSET', 'ADC_U_LOAD_GAIN', 'ADC_U_LOAD_OFFSET',
                 'ADC_U_REGULATOR_GAIN', 'ADC_U_REGULATOR_OFFSET',
                 'ADC_I_MON_GAIN', 'ADC_I_MON_OFFSET', 'DAC_CURRENT_GAIN', 'DAC_CURRENT_OFFSET']
    groups = [group1, group2, group3, group4, group5, group6]
    with PdfPages(f'../data/channel_grouping_boxplots.pdf') as pdf:
        for n in range(10):
            print(f"Plotting {plotnames[n]}...")
        #with PdfPages(f'/home/silab44/Desktop/Frederik/Plots/boxplots_grouped_{vars[n]}.pdf') as pdf:
            fig, ax = plt.subplots(2, 3)
            fig.set_figheight(5)
            fig.set_figwidth(11)
            st = fig.suptitle(plotnames[n].replace("_", " "))
            for plot in range(6):
                x = np.array(groups[plot])
                xt = []
                for i in range(len(x)):
                    #print(channel_names[int(xt[i])])
                    s1 = str(x[i])
                    s2 = channel_names[int(x[i])]
                    if plot == 0: # or plot == 5:
                        if i%2 == 1:
                            xt.append(''.join((s1, "\n\n", s2)))
                        else:
                            xt.append(''.join((s1, "\n", s2)))
                    else:
                        xt.append(''.join((s1, "\n", s2)))
                y = np.zeros(len(data[names[0]][mask]))
                list = []
                for g in groups[plot]:
                    arg = int(float(g))
                    y = data[names[4 + arg * 10 + n]][mask]
                    list.append(y)
                if plot < 3:
                    ax[0, plot % 3].boxplot(list)
                    ax[0, plot % 3].set_xticklabels(xt)
                    #ax2 = ax[0, plot%3].twiny()
                    #ax2.set_xticklabels(x)
                else:
                    ax[1, plot % 3].boxplot(list)
                    ax[1, plot % 3].set_xticklabels(xt)
                    #ax2 = ax[1, plot%3].twiny()
                    #ax2.set_xticklabels(x)
                # secret techniques
                #plt.setp(ax[0,0].get_xticklabels(), rotation=90, horizontalalignment='center')
                #plt.setp(ax[0, 2].get_xticklabels(), rotation=90, horizontalalignment='center')
                #plt.setp(ax[1, 2].get_xticklabels(), rotation=90, horizontalalignment='center')
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            pdf.savefig()
            plt.close()

def calc_mean(config,group,name,ASIC,writer):
    means = []
    #print(group)
    for const in range(10):
        #print(vars[const])
        values = np.zeros(len(group))
        for x in range(len(values)):
            values[x] = config[f'{group[x]}'][f'{vars[const]}_{group[x]}']
            #print(group[x],values[x])
        if ASIC == True:
            if const == 6 or const == 7:
                #means.append(np.mean(values[0,2:]))
                values = np.delete(values,1)
                print(values)
            if const == 8 or const == 9:
                values = np.delete(values,(1,6))
                print(values)
                #means.append(np.mean(values[0,2:-1]))
        means.append(np.mean(values))
        #print(f'{vars[const]}:\nmean: {mean}')
    means.insert(0, name)
    means.insert(1, 'mean')
    writer.writerow(means)

def calc_std(config,group,name,ASIC,writer):
    stds = []
    for const in range(10):
        values = np.zeros(len(group))
        for x in range(len(values)):
            values[x] = config[f'{group[x]}'][f'{vars[const]}_{group[x]}']
        if ASIC == True:
            if const == 6 or const == 7:
                #means.append(np.mean(values[0,2:]))
                values = np.delete(values,1)
            if const == 8 or const == 9:
                values = np.delete(values,(1,6))
                #means.append(np.mean(values[0,2:-1]))
        stds.append(np.mean(values))
        #print(f'{vars[const]}:\nstd: {mean}')
    stds.insert(0, name)
    stds.insert(1, 'std')
    writer.writerow(stds)

def calculate_valid_constants():
    # calculate mean and std via arithmetic mean of database.ini ranges
    config_mean, config_std = configparser.ConfigParser(), configparser.ConfigParser()
    config_mean.read('./../data/database.ini')
    config_std.read('./../data/database_std.ini')
    header_names = vars.copy()
    header_names.insert(0,'Group')
    header_names.insert(1,'values')
    with open(f'../data/valid_ranges.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_names)
        #writer = csv.DictWriter(csvfile, fieldnames=header_names)
        #writer.writeheader()
        calc_mean(config_mean,group1,'ASIC',True,writer)
        calc_std(config_std,group1,'ASIC',True,writer)
        calc_mean(config_mean,group2,'ccg',False,writer)
        calc_std(config_std,group2,'ccg',False,writer)
        calc_mean(config_mean, group3, 'gate', False, writer)
        calc_std(config_std, group3, 'gate', False, writer)
        calc_mean(config_mean, group4, 'clear', False, writer)
        calc_std(config_std, group4, 'clear', False, writer)
        calc_mean(config_mean, group5, 'sw', False, writer)
        calc_std(config_std, group5, 'sw', False, writer)
        calculate_range_total(writer)

def calculate_range_total(writer):
    # calculate constants directly from database
    writer.writerow(['Calculated from database'])
    # calculate mean and std directly from database.csv
    data = pd.read_csv('../data/database.csv')
    valid = data['used_for_range'] == 'yes'

    # special treatment for group 1:
    stds = np.zeros(10)
    means = np.zeros(10)
    help_group = group1.copy()
    for const in range(10):
        values = np.empty(1)
        for channel in help_group:
            add = data[names[4 + channel * 10 + const]][valid]
            #print(names[4 + channel * 10 + const])
            # print(add)
            values = np.concatenate((values, add), axis=None)
        # print(values.shape)
        means[const] = np.mean(values[1:])
        stds[const] = np.std(values[1:])
        if const == 5:
            help_group = np.delete(help_group,1)
        if const == 7:
            help_group = np.delete(help_group,5)
    #print(means)
    list = ['ASIC','mean'] + means.tolist()
    writer.writerow(list)
    list = ['ASIC','std'] + stds.tolist()
    writer.writerow(list)
    ini_list = list[2:].copy()
    print(ini_list)

    # group 2 to 5:
    index = 0
    for group in [group2, group3, group4, group5]:
        stds = np.zeros(10)
        means = np.zeros(10)
        for const in range(10):
            values = np.empty(1)
            for channel in group:
                add = data[names[4 + channel * 10 + const]][valid]
                #print(names[4+channel*10+const])
                #print(add)
                values = np.concatenate((values,add),axis=None)
            #print(values.shape)
            means[const] = np.mean(values[1:])
            stds[const] = np.std(values[1:])
        head = ['ccg','gate','clear','sw']
        list = [head[index],'mean'] + means.tolist()
        writer.writerow(list)
        list = [head[index],'std'] + stds.tolist()
        writer.writerow(list)
        index += 1
        print(means)

def constants_variance_grouped():
    data = pd.read_csv('../data/valid_ranges.csv')
    with PdfPages(f'../data/constants_variance_grouped.pdf') as pdf:
        x,y = np.arange(0,5,1), np.zeros(5)
        group_names = data['Group'][11::2]
        for const in range(10):
            values = data[vars[const]]
            plt.figure()
            _, ax = plt.subplots()
            for group in range(5):
                y[group] = np.abs(values[12+group*2]/values[11+group*2])
            plt.bar(x,y)
            plt.title(vars[const])
            plt.ylabel('Constants variance')
            ax.set_xticks(x)
            ax.set_xticklabels(group_names)
            pdf.savefig()
            plt.close()
            print(vars[const])
            print(y)

def constants_variance_single():
    config_vals = configparser.ConfigParser()
    config_vals.read(f'../data/database.ini')
    config_errs = configparser.ConfigParser()
    config_errs.read(f'../data/database_std.ini')
    plotnames=['DAC_VOLTAGE_GAIN','DAC_VOLTAGE_OFFSET','ADC_U_LOAD_GAIN','ADC_U_LOAD_OFFSET','ADC_U_REGULATOR_GAIN','ADC_U_REGULATOR_OFFSET',
           'ADC_I_MON_GAIN','ADC_I_MON_OFFSET','DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET']
    with PdfPages(f'../data/Constants_variance_single.pdf') as pdf:
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

def constants_variance_total():
    data = pd.read_csv('../data/final_ranges.csv',header=None)
    data = data.T
    data.drop(0,axis=0)
    print(data)
    with PdfPages(f'../data/final_ranges.pdf') as pdf:
        x,y = np.arange(0,10,1), np.zeros(10)
        group_names = data[0][2:12].tolist()
        for const in range(10):
            values = data[2*const+2][2:]
            plt.figure()
            _, ax = plt.subplots()
            for group in range(10):
                y[group] = values[group+2]
            plt.bar(x,y)
            plt.title(vars[const].replace("_"," "))
            if const % 2 == 0:
                plt.ylabel('Relative range differences')
            else:
                plt.ylabel('Absolute range differences')
            ax.set_xticks(x)
            xt = group_names[:5]+['source','poly','HV','guard','bulk']
            ax.set_xticklabels(xt)
            pdf.savefig()
            plt.close()

def final_ranges_to_ini():
    # for the groups:
    config = configparser.ConfigParser()
    data = pd.read_csv('../data/valid_ranges.csv')
    config_dict = {}
    for gr in range(5):
        print(data["Group"][11+2*gr])
        for n in range(10):
            config_dict[vars[n]] = data[vars[n]][11+2*gr]
            config_dict[f'{vars[n]}_diff'] = data[vars[n]][12+2*gr]
        config[data["Group"][11+2*gr]] = config_dict

    #database = pd.read_csv('../data/database.csv')
    config_channel = configparser.ConfigParser()
    config_channel.read('../data/database.ini')
    config_channel_std = configparser.ConfigParser()
    config_channel_std.read('../data/database_std.ini')

    # for channel 1 and 7 in the ASIC group
    config_dict = {}
    for name in ['DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET','ADC_I_MON_GAIN','ADC_I_MON_OFFSET']:
        config_dict[name] = config_channel['1'][f'{name}_1']
        config_dict[f'{name}_diff'] = config_channel_std['1'][f'{name}_1']
    config['1'] = config_dict
    config_dict = {}
    for name in ['DAC_CURRENT_GAIN','DAC_CURRENT_OFFSET']:
        config_dict[name] = config_channel['7'][f'{name}_7']
        config_dict[f'{name}_diff'] = config_channel_std['7'][f'{name}_7']
    config['7'] = config_dict

    # for the non grouped channels:
    for channel in [5,12,13,14,15]:
        config_dict = {}
        for name in vars:
            config_dict[name] = config_channel[f'{channel}'][f'{name}_{channel}']
            config_dict[f'{name}_diff'] = config_channel_std[f'{channel}'][f'{name}_{channel}']
        config[channel] = config_dict

    with open(f'../data/final_ranges.ini', 'w') as configfile:
        config.write(configfile)

def final_ranges_relative():
    config = configparser.ConfigParser()
    config.read('../data/final_ranges.ini')
    d = vars.copy()
    for l in range(len(d)):
        d.insert(2*l+1,f'{vars[l]}_DIFF')
    df = pd.DataFrame(data=d, columns=['Group'])
    #print(df)
    groups = ['ASIC','ccg','gate','sw','clear','5','12','13','14','15']
    for name in groups:
        list = []
        for item in d[:]:
            list.append(config[name][item])
        df[name] = list
    df['1'] = df['ASIC']
    df['7'] = df['ASIC']
    #df.rename(columns={'ASIC': '1', 'ASIC': '7'})
    df['1'][12] = config['1']['ADC_I_MON_GAIN']
    df['1'][13] = config['1']['ADC_I_MON_GAIN_DIFF']
    df['1'][14] = config['1']['ADC_I_MON_OFFSET']
    df['1'][15] = config['1']['ADC_I_MON_OFFSET_DIFF']
    df['1'][16] = config['1']['DAC_CURRENT_GAIN']
    df['1'][17] = config['1']['DAC_CURRENT_GAIN_DIFF']
    df['1'][18] = config['1']['DAC_CURRENT_OFFSET']
    df['1'][19] = config['1']['DAC_CURRENT_OFFSET_DIFF']
    df['7'][16] = config['7']['DAC_CURRENT_GAIN']
    df['7'][17] = config['7']['DAC_CURRENT_GAIN_DIFF']
    df['7'][18] = config['7']['DAC_CURRENT_OFFSET']
    df['7'][19] = config['7']['DAC_CURRENT_OFFSET_DIFF']
    #df = df.T
    print(df)

    # get relative differences:
    groups.append('1')
    groups.append('7')
    for name in groups:
        for n in range(5):
            df[name][1+ n*4] = np.abs(float(df[name][1+n*4])/float(df[name][n*4]))
    df.to_csv('../data/final_ranges.csv')

def compare_values_to_new_range():
    group = 'ccg'
    channels = np.array([8,9,10,11])
    data = pd.read_csv('../data/database.csv')
    valid = data['used_for_range'] == 'yes'
    config_range = configparser.ConfigParser()
    config_range.read('../data/final_ranges.ini')
    with PdfPages(f'../data/compare_values_to_range_{group}.pdf') as pdf:
        for const in vars:
            plt.figure()
            _,ax = plt.subplots()
            env = 5
            range_high, range_low = float(config_range[f'{group}'][const]) + env*float(config_range[f'{group}'][f'{const}_diff']), float(config_range[f'{group}'][const]) - env*float(config_range[f'{group}'][f'{const}_diff'])
            plt.ylim(range_low - 2*float(config_range[f'{group}'][f'{const}_diff']), range_high+2*float(config_range[f'{group}'][f'{const}_diff']))
            plt.fill_between(x=np.array([np.amin(channels)-0.5,np.amax(channels)+0.5]), y1=range_low, y2=range_high, alpha = 0.6,label='valid range')
            for channel in channels:
                values = data[f'{const}_{channel}'][valid]
                x = np.ones_like(values) * channel
                plt.scatter(x, values,label=channel_names[channel])
            plt.title(const)
            plt.xlabel('Channels')
            plt.ylabel('Constant value')
            ax.set_xticks(channels)
            ax.set_xticklabels(channels)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    #check_all()
    ######
    # Note: 60 out of 201 calibrations are successful with the new range!
    ######
    #boxplot_per_constant()
    #boxplot_per_channel()
    #channel_grouping()
    #channel_grouping_by_board()
    grouping_boxplots()
    #calculate_valid_constants()
    #final_ranges_to_ini()
    #constants_variance_grouped()
    #final_ranges_relative()
    #constants_variance_total()
    #compare_values_to_new_range()