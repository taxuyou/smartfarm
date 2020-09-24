
import os

import pandas as pd
import numpy as np
import tensorflow as tf

__all__ = ['mlp_dataloader']

def mlp_dataloader(args):
    path = args.directory
    mode = args.target

    print("target: growth")
    grw = args.train_data.grw
    env = args.train_data.env

    df_grw = pd.read_excel(path+grw)
    df_grw['날짜'] = df_grw.apply(
        lambda row: int('2018'+row['주차'].split('(')[1].replace(')','').replace('/','')),axis=1)
    df_grww = df_grw.copy()

    # Preprocess Growth Data
    grw_date = df_grw['날짜'].unique()
    grw_date = pd.DataFrame(grw_date)

    df_grw = df_grw.set_index(['샘플번호','날짜'])
    df_grw = df_grw.sort_index()
    df_grw = df_grw.reindex(columns=['초장(cm)','줄기굵기(mm)','잎길이(cm)','잎폭(cm)','leaf_area'])

    for [samp,date],rows in df_grw.iterrows():
        df_grw.loc[samp,:].replace(to_replace=0.0,method='ffill',inplace=True)
    df_grw_label = df_grww.shift(-1).reindex(columns=['주간생육길이(cm)','줄기굵기(mm)','잎길이(cm)','잎폭(cm)','leaf_area'])
    df_grw_label.rename(columns={'줄기굵기(mm)':'d줄기굵기(mm)','잎길이(cm)':'d잎길이(cm)','잎폭(cm)':'d잎폭(cm)','leaf_area':'dleaf_area'},inplace=True)
    df_growth_total = pd.concat([df_grw.reset_index(drop=True),df_grw_label.reset_index(drop=True)], axis=1)
    
    # process Environmnet
    df_env = pd.read_excel(path + env)
    df_env.drop(['low_temperature','high_temperature','suitable_temperature',
        'daytime_temperature', 'nighttime_temperature', 'afternoon_temperature',
        'suitable_temperature_time', 'suitable_humidity_time', 
        ],inplace=True,axis=1)

    date = pd.DataFrame(df_env['날짜'])
    df_grw = df_grw.reset_index(drop=True)
    startdate = df_grww['날짜'][0]
    enddate = df_grww['날짜'].iloc[-1]
    startindex = date.index[date['날짜']==startdate][0]

    grw_idx = []
    for idx, row in grw_date.iterrows():
        a = date.index[date['날짜']==row[0]][0]
        grw_idx.append(a)
        
    env_idx = []
    for x in grw_idx:
        arr = [x-14,x-13,x-12,x-11,x-10,x-9,x-8,x-7,x-6,x-5,x-4,x-3,x-2,x-1,x,x+1,x+2,x+3,x+4,x+5,x+6]
        # arr = [x,x+1,x+2,x+3,x+4,x+5,x+6]
        env_idx.append(arr)

    df_feat = []
    for x in range(len(env_idx)):
        a = df_env.iloc[env_idx[x]].stack().T.values
        df_feat.append(a)
    df_feat = pd.DataFrame(df_feat)


    env_cols = []
    for y in range(21):
        for x in range(len(df_env.columns)):
            env_cols.append(df_env.columns[x]+'_' + str(y))
    df_feat.columns = env_cols 

    df_feat = pd.DataFrame(df_feat)
    df_label = df_growth_total.reset_index().set_index(['index'])
    df_fets = pd.concat([df_feat]*16)

    # print(df_fets)
    # df_target = df_label.drop(['초장(cm)','줄기굵기(mm)','잎길이(cm)','잎폭(cm)','leaf_area'],axis=1)
    df_target = df_label
    data = pd.concat([df_fets.reset_index(drop=True),df_target.reset_index(drop=True)],axis=1)
    data = data[data['날짜_13'] != startdate]
    data = data[data['날짜_13'] != startdate+7]

    data = data[data['날짜_13'] != enddate-7]
    data = data[data['날짜_13'] != enddate]

    data = data.drop(['날짜_0','날짜_1','날짜_2','날짜_3','날짜_4','날짜_5','날짜_6','날짜_7','날짜_8','날짜_9','날짜_10','날짜_11','날짜_12','날짜_14','날짜_15','날짜_16','날짜_17','날짜_18','날짜_19','날짜_20','날짜_6'],axis=1)
    dataset = data.interpolate()
        

    ds = dataset
    return ds





def process_product(ds):
    print('processing product')

def first_step_dataset(ds):
    print('first step dataset')

def second_step_dataset(ds):
    print('seoncd step dataset')