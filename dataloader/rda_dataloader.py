'''
rad: Rural Development Administration (농촌진흥청)
'''

import os
from datetime import datetime
from math import sqrt

import numpy as np
from numpy import concatenate
from numpy import asarray

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from utils.plotutils import draw_rda_dataframe


__all__ = ['data_transform', 'split_dataset', 'preprocessing_3rd_data', 'get_dt_dataset', 'get_bl_dataset']


'''

'''
def preprocessing_3rd_data(args):

    preproc_df = None

    if os.path.isfile(args.data.path+'/'+args.data.raw_data.names[0]) == True:

        if os.path.isfile(args.data.path+'/'+args.data.preprocessed_data.names[0]) == False:

            # load raw data without parsing
            # type: ExcelFile
            xls = pd.ExcelFile(args.data.path+'/'+args.data.raw_data.names[0])
            print(type(xls))

            # get a list of sheet names
            sheets = xls.sheet_names
            print(sheets)

            # parse and read the 1st sheet
            # type: dataframe
            sheet1 = xls.parse(sheets[1])
            print(sheet1)
            print(type(sheet1))

            # Read all sheets and store it in a dictionary.
            # type: dict
            sheet_to_df_map = {}
            for sheet_name in xls.sheet_names:
                sheet_to_df_map[sheet_name] = xls.parse(sheet_name)
            #print(sheet_to_df_map)
            print(type(sheet_to_df_map))

            key_list = list(sheet_to_df_map.keys())
            val_list = list(sheet_to_df_map.values())
            print(type(key_list))
            print(sorted(key_list))
            print("\n")
            print(type(val_list))
            print("rows: %d, columns: %d" %(len(val_list), len(val_list[0])))
            print(val_list[0].head(5))
            print("\n")

            # make each sheet to pandas dataframe
            env_df = pd.DataFrame()
            excluded_sheets = ['항목', '생육', '생산량']
            columns = list(range(3, 9))

            for sheet in sorted(key_list):
                if sheet not in excluded_sheets:
                    # convert a list to a dataframe
                    df = pd.DataFrame(sheet_to_df_map.get(sheet))
                    print("\n\nsheet name:", sheet)
                    print(df.columns)
                    print(df.shape)
                    
                    # plot
                    draw_rda_dataframe(dataframe=df, groups=columns, sheet=sheet)
                    
                    # merge environment sheets to one
                    env_df = env_df.append(df)

            print(env_df.shape)
            for i in range(23, len(env_df), 24):
                print(i, env_df.values[i].astype('int32'))

            '''
            merge modified environment dataframe to a new df
            '''

            # environment dataframe with one week interval
            env_week_df = pd.DataFrame()

            # sampling rate is one week (24hrs x 7 days)
            sampling_rate = 24*7

            print(env_df.columns)

            columns = list(range(3, 9))
            for column in columns:
                col_name = str(env_df.columns[column])
                print("column name:", col_name)
                
                if col_name != 'acSlrdQy':
                    # average, maximum, and minimum values for a week
                    val_avg = [(env_df[col_name].values[i:i+(sampling_rate-1)].mean()).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
                    val_med = [np.median(env_df[col_name].values[i:i+(sampling_rate-1)]).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
                    val_min = [(env_df[col_name].values[i:i+(sampling_rate-1)].min()).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
                    val_max = [(env_df[col_name].values[i:i+(sampling_rate-1)].max()).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
            
                    # convert ndarray to series
                    sr_avg = pd.Series(val_avg)
                    sr_med = pd.Series(val_med)
                    sr_min = pd.Series(val_min)
                    sr_max = pd.Series(val_max)
                    env_week_df.loc[:, col_name+str('_avg')] = sr_avg
                    env_week_df.loc[:, col_name+str('_med')] = sr_med
                    env_week_df.loc[:, col_name+str('_min')] = sr_min
                    env_week_df.loc[:, col_name+str('_max')] = sr_max
                    
                elif col_name == 'acSlrdQy':
                    # average value and accumulated value for a week
                    val_avg = [(env_df[col_name].values[i:i+(sampling_rate-1)].mean()).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
                    val_acc = [(env_df[col_name].values[i:i+(sampling_rate-1)].sum()).astype('float64') for i in range(0, len(env_df.values), sampling_rate)]
                    sr_avg = pd.Series(val_avg)
                    sr_acc = pd.Series(val_acc)
                    env_week_df.loc[:, col_name+str('_avg')] = sr_avg
                    env_week_df.loc[:, col_name+str('_acc')] = sr_acc
                
            # read growth and yield sheets and convert them to dataframe
            grow_df = pd.DataFrame()
            yield_df = pd.DataFrame()
            target_sheets = ['생육', '생산량']
            columns = []

            for sheet in sorted(key_list):
                if sheet in target_sheets:
                    # convert a list to a dataframe
                    df = pd.DataFrame(sheet_to_df_map.get(sheet))
                    print("\n\nsheet name:", sheet)
                    print(df.columns)
                    print(df.shape)
                    
                    # get a dataframe from each sheet and plot it
                    data_type = None
                    if sheet == target_sheets[0]:
                        columns = list(range(4, 16))
                        grow_df = df
                    elif sheet == target_sheets[1]:
                        columns = [3, 5]
                        yield_df = df
                    #draw_rda_dataframe(dataframe=df, groups=columns, sheet=sheet)
                    
            print(grow_df.shape)
            print(yield_df.shape)

            '''
            modify growth dataframe
            '''
            grow_week_df = pd.DataFrame()

            # sampling rate is one week
            sampling_rate = 2

            columns = list(range(4, 16))
            for column in columns:
                col_name = str(grow_df.columns[column])
                print("column name:", col_name)
                
                # average value for a week
                val_avg = [(grow_df[col_name].values[i:i+(sampling_rate-1)].mean()).astype('float64') for i in range(0, len(grow_df.values), sampling_rate)]
                print(len(val_avg), val_avg)
                
                sr_avg = pd.Series(val_avg)
                grow_week_df.loc[:, col_name+str('_avg')] = sr_avg

            # remove '수확수' column from growth dataframe
            grow_week_df = grow_week_df.drop('수확수_avg', 1)    
            
            # modify yield dataframe
            columns = yield_df.columns
            print(columns)

            for i in range(0, 6):
                if i != 3 and i != 5:
                    yield_df = yield_df.drop(columns[i], axis=1)

            yield_week_df = yield_df

            # merge growth and yield dataframes with environment dataframe
            # note - env_week_df is NOT appended
            merged_df = env_week_df.join(grow_week_df)
            merged_df = merged_df.join(yield_week_df)

            # save the merged dataframe to a csv file and read the file for test purpose
            merged_df.to_csv(args.data.path+'/'+args.data.preprocessed_data.names[0], index=True, header=True)
            
            # check
            statinfo_raw = os.stat(args.data.path+'/'+args.data.raw_data.names[0])
            statinfo_df = os.stat(args.data.path+'/'+args.data.preprocessed_data.names[0])
            print("\nraw data.xlsx - size: {} bytes".format(statinfo_raw.st_size))
            print("\ndataframe.csv - size: {} bytes".format(statinfo_df.st_size))

            preproc_df = merged_df
        
        else:
            preproc_df = read_csv(args.data.path+'/'+args.data.preprocessed_data.names[0], header=0, index_col=0)
            print(preproc_df.shape)

    else:
        print("Raw data is missing - %s/%s" % (args.data.path, args.data.raw_data.names[0]))

    return preproc_df


'''
transform a time series dataset into a supervised learning dataset
'''
def data_transform(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    # return supervised dataframe
    return agg


'''
split dataset into train and test
'''
def split_dataset(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

       
'''
ooo
'''
def get_dt_dataset(args):

    data = None
    dataset = []

    preproc_df = preprocessing_3rd_data(args)
    
    if preproc_df.empty == False:
        # load the dataset
        values = preproc_df.values

        # transform the time series data into supervised learning
        lookback = int(args.data.supervised_data.lookback)
        pred_out = int(args.data.supervised_data.out)
        data = data_transform(values, n_in=lookback, n_out=pred_out)
        data = data.reset_index(drop=True)

        # drop 
        features = int(preproc_df.shape[1])
        drop_columns = list(range(features * lookback, features * (lookback+1)-1))
        data.drop(data.columns[drop_columns], axis=1, inplace=True)

        # check the preprocessed dataframe
        print(data)

        # save
        data.to_csv(args.data.path+'/'+args.data.supervised_data.names[0], index=True, header=True)
 
        # use all samples for training
        if int(args.train.test_samples) == 0:
            train_X, train_y = data.values[:, :-1], data.values[:, -1]
            dataset.append(train_X)
            dataset.append(train_y)
            print(train_X.shape)
            print(train_y.shape)
        # split samples to train and test datasets
        else:
            test_num = int(args.train.test_samples)
            train, test = split_dataset(data.values, test_num)
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            dataset.append(train_X)
            dataset.append(train_y)
            dataset.append(test_X)
            dataset.append(test_y)
            print(train.shape)
            print(test.shape)
            print(train_X.shape)
            print(train_y.shape)
            print(test_X.shape)
            print(test_y.shape)

    return preproc_df, data, dataset


'''
ooo
'''
def get_bl_dataset(args):

    data = None
    dataset = []

    preproc_df = preprocessing_3rd_data(args)
    
    if preproc_df.empty == False:
        # load the dataset
        values = preproc_df.values

        # scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        scaled = scaled.astype('float32')

        # transform the time series data into supervised learning
        lookback = int(args.data.supervised_data.lookback)
        pred_out = int(args.data.supervised_data.out)
        data = data_transform(values, n_in=lookback, n_out=pred_out)
        data = data.reset_index(drop=True)

        # drop 
        features = int(preproc_df.shape[1])
        drop_columns = list(range(features * lookback, features * (lookback+1)-1))
        data.drop(data.columns[drop_columns], axis=1, inplace=True)

        # save
        data.to_csv(args.data.path+'/'+args.data.supervised_data.names[0], index=True, header=True)
 
        # check the preprocessed dataframe
        print(data)

        # split into train and test sets
        train, test = split_dataset(data.values, n_test=12)
        print(train.shape, test.shape)

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        print(train_X.shape, train_y.shape)
        print(test_X.shape, test_y.shape)

        # reshape 2D input shape (samples, timesteps*features) 
        # to 3D shape (samples, timesteps, features)
        # @params:
        #       number of samples
        #       number of lookback timesteps
        #       number of features
        train_X = train_X.reshape((train_X.shape[0], lookback, int(train_X.shape[1]/lookback)))
        test_X = test_X.reshape((test_X.shape[0], lookback, int(test_X.shape[1]/lookback)))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        print(train_y)
        print(test_y)

        # dataset
        dataset.append(train_X)
        dataset.append(train_y)
        dataset.append(test_X)
        dataset.append(test_y)
        
    return preproc_df, data, dataset, scaler