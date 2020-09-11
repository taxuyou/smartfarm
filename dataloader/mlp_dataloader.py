
import os

import pandas as pd
import numpy as np
import tensorflow as tf

__all__ = ['mlp_dataloader']

def mlp_dataloader(args):
    train_data_list = []
    
    path = args.data.train_data.path
    growth = args.data.train_data.growth
    env = args.data.train_data.env
    product_1 = args.data.train_data.product_1
    product_2 = args.data.train_data.product_2
    product_3 = args.data.train_data.product_3
    product_4 = args.data.train_data.product_4

    mode = args.target
    dataset = []
    ############################################################################
    # Read Data from datapaths (Growth, Env, Product)
    ############################################################################

    

    df_grw = pd.read_excel(path + growth)

    df_grw['날짜'] = df_grw.apply(
        lambda row: int('2018'+row['주차'].split('(')[1].replace(')','').replace('/','')),axis=1)
    df_grww = df_grw.copy()

    # Preprocess Growth Data
    grw_date = df_grw['날짜'].unique()
    grw_date = pd.DataFrame(grw_date)

    df_grw = df_grw.set_index(['샘플번호','날짜'])
    df_grw = df_grw.sort_index()
    df_grw = df_grw.reindex(columns=['초장(cm)','줄기굵기(mm)','잎길이(cm)','잎폭(cm)','leaf_area'])


    if mode == "product":

        df_p1 = pd.read_excel(path + product_1)
        df_p1['날짜'] = df_p1.apply(lambda row: int(row['날짜'].strftime('%Y%m%d')),axis=1)
        df_p1.drop(df_p1.index[df_p1['샘플번호'] == '2-4-10-9'])
        df_p1['샘플번호'] = df_p1.apply(lambda row: int(row['샘플번호'].split('-')[3]),axis=1)


        df_p2 = pd.read_excel(path + product_2)
        df_p2['날짜'] = df_p2.apply(lambda row: int(row['날짜'].strftime('%Y%m%d')),axis=1)
        df_p2.drop(df_p2.index[df_p2['샘플번호'] == '2-4-10-9'])
        df_p2['샘플번호'] = df_p2.apply(lambda row: int(row['샘플번호'].split('-')[3]),axis=1)


        df_p3 = pd.read_excel(path + product_3)
        df_p3['날짜'] = df_p3.apply(lambda row: int(row['날짜'].strftime('%Y%m%d')),axis=1)
        df_p3.drop(df_p3.index[df_p3['샘플번호'] == '2-4-10-9'])
        df_p3['샘플번호'] = df_p3.apply(lambda row: int(row['샘플번호'].split('-')[3]),axis=1)

        df_p4 = pd.read_excel(path + product_4)
        df_p4['날짜'] = df_p4.apply(lambda row: int(row['날짜'].strftime('%Y%m%d')),axis=1)

        df_p4.drop(df_p4.index[df_p4['샘플번호'] == '2-4-10-9'])
        df_p4['샘플번호'] = df_p4.apply(lambda row: int(row['샘플번호'].split('-')[3]),axis=1)

        df_p1.sort_values(['샘플번호','날짜'],inplace=True)
        df_p2.sort_values(['샘플번호','날짜'],inplace=True)
        df_p3.sort_values(['샘플번호','날짜'],inplace=True)
        df_p4.sort_values(['샘플번호','날짜'],inplace=True)


        df_p1.drop(['날짜','샘플번호'],axis=1,inplace=True)
        df_p2.drop(['날짜','샘플번호'],axis=1,inplace=True)
        df_p3.drop(['날짜','샘플번호'],axis=1,inplace=True)
        df_p4.drop(['날짜','샘플번호'],axis=1,inplace=True)

        df_label = pd.concat([df_p1.reset_index(drop=True),df_p2.reset_index(drop=True),df_p3.reset_index(drop=True),df_p4.reset_index(drop=True)],axis=1)
        dataset = pd.concat([df_grw.reset_index(drop=True),df_label.reset_index(drop=True)], axis=1)\

    else: # Growth mode
        # Process Growth label for growth mode
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
            arr = [x,x+1,x+2,x+3,x+4,x+5,x+6]
            env_idx.append(arr)

        df_feat = []
        for x in range(len(env_idx)):
            a = df_env.iloc[env_idx[x]].stack().T.values
            df_feat.append(a)

        df_feat = pd.DataFrame(df_feat)
        df_label = df_growth_total.reset_index().set_index(['index'])
        df_fets = pd.concat([df_feat]*16)

        # Create Dataset
        env_grw = pd.concat([df_fets.reset_index(drop=True),df_label.reset_index(drop=True)], axis=1)

        env_grw = env_grw[env_grw[0] != startdate]
        env_grw = env_grw[env_grw[0] != startdate+7]

        env_grw = env_grw[env_grw[0] != enddate-7]
        env_grw = env_grw[env_grw[0] != enddate]

        dataset = env_grw.drop([0,70,140,210,280,350,420],1)

    ds = dataset
    return ds




def process_product(ds):
    print('processing product')

def first_step_dataset(ds):
    print('first step dataset')

def second_step_dataset(ds):
    print('seoncd step dataset')