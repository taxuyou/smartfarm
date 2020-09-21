import os
import numpy as np
import pandas as pd
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, default='./config.json')
conf = parser.parse_args()

def HD_cumulative_time(df=None):
    if df is None:
        raise ValueError(df)
    
    am_df = df[df['시간'] < 120000]
    am_suitable_df = am_df[(am_df['HD'] >= 2.8) & (am_df['HD'] <= 6)]
    am_unsuitable_df = am_df[am_df['HD'] < 1.1]
    
    sun_df = df[(df['시간'] >= 60000) & (df['시간'] <= 190000)]
    sun_suitable_df = sun_df[(sun_df['HD'] >= 2.8) & (sun_df['HD'] <= 6)]
    
    all_day_suitable_df = df[(df['HD'] >= 2.8) & (df['HD'] <= 6)]
    
    pm_df = df[df['시간'] >= 120000]
    pm_unsuitable_df = pm_df[pm_df['HD'] >= 15]
    
    HD_cumulative_list = [len(am_suitable_df)/60, len(sun_suitable_df)/60, len(all_day_suitable_df)/60,
                          len(pm_unsuitable_df)/60, len(am_unsuitable_df)/60]
    
    HD = pd.DataFrame([HD_cumulative_list])
    HD.columns = ['am_suitable_HD', 'sun_suitable_HD', 'day_suitable_HD', 'pm_unsuitable_HD', 'am_unsuitable_HD']
    
    return HD

def temperature_cumulative_time(df=None):
    if df is None:
        raise ValueError(df)
    
    low_temp_df = df[df['실내온도'] <= 12]
    high_temp_df = df[df['실내온도'] >= 30]
    
    suitable_df = df[(df['실내온도'] >= 18) & (df['실내온도'] <= 20)]
    
    temp_cumulative_list = [len(low_temp_df)/60, len(high_temp_df)/60, len(suitable_df)/60]
    
    temp = pd.DataFrame([temp_cumulative_list])
    temp.columns = ['low_temperature', 'high_temperature', 'suitable_temperature']
    
    return temp

def temperature_average(df=None):
    if df is None:
        raise ValueError(df)
    
    daytime_df = df[(df['시간'] >= 60000) & (df['시간'] <= 190000)]
    daytime_temp = daytime_df['실내온도'].mean()
    
    nighttime_df = df[(df['시간'] < 60000) | (df['시간'] > 190000)]
    nighttime_temp = nighttime_df['실내온도'].mean()
    
    afternoon_df = df[(df['시간'] >= 120000) & (df['시간'] <= 190000)]
    afternoon_temp = afternoon_df['실내온도'].mean()
    
    temp_avg_list = [daytime_temp, nighttime_temp, afternoon_temp]
    temp = pd.DataFrame([temp_avg_list])
    temp.columns = ['daytime_temperature', 'nighttime_temperature', 'afternoon_temperature']
    
    return temp

def suitable_info(df=None):
    if df is None:
        raise ValueError(df)
    
    suitable_humid = df[(df['실내습도'] >= 65) & (df['실내습도'] <= 85)]
    
    day_temp_lowerbound = df['실내온도'].min() - 16
    day_temp_upperbound = df['실내온도'].max() - 26
    
    sunrise = df[(df['시간'] >= 50000) & (df['시간'] <= 70000)]
    sunrise_diff = sunrise['실내온도'].max() - sunrise['실내온도'].min()
    
    sunset = df[(df['시간'] >= 180000) & (df['시간'] <= 200000)]
    sunset_diff = sunset['실내온도'].max() - sunset['실내온도'].min()
    
    suitable_list = [len(suitable_humid)/60, sunrise_diff, sunset_diff, 
                     day_temp_lowerbound, day_temp_upperbound]
    
    suitable = pd.DataFrame([suitable_list])
    suitable.columns = ['suitable_humidity_time', 'sunrise_diff', 'sunset_diff',
                       'suitable_temperature_diff_lowerbound', 'suitable_temperature_diff_upperbound']
    return suitable

def env_data_preprocessing(args):
    scr = args.env.src_path
    files = args.env.names
    save_name = os.path.join(args.env.dest_path, args.env.save_name)

    origin_headers = ['날짜', '시간', '실내온도', '실내습도', 'dew point (Td)', 'HD', 'Height above sea level', 'atmospheric pressure',
           'pws', 'W (moisture content)', 'h (enthalpy)', 'v (specific volume)', 'rho (density)', 
           'pw (Partial pressure) water vapor', 'Absolute Humidity (AH)', 'absoluteAH']

    env_df = pd.DataFrame()
    for f in files:
        file_path = os.path.join(scr, f)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        sheet_df.columns = origin_headers
        env_df = pd.concat([env_df, sheet_df])
    
    new_env_df = env_df.drop(env_df.index[0])
    new_env_df['날짜'] = new_env_df['날짜'].astype(float)
    new_env_df = new_env_df.drop(columns=['Height above sea level'])
    new_env_df = new_env_df.dropna(axis=0)
    new_env_df['시간'] = new_env_df['시간'].astype(float)

    headers = ['날짜', '시간','실내온도', '실내습도', 'dew point (Td)', 'HD', 'atmospheric pressure',
       'pws', 'W (moisture content)', 'h (enthalpy)', 'v (specific volume)',
       'rho (density)', 'pw (Partial pressure) water vapor', 'Absolute Humidity (AH)', 'absoluteAH']

    std_headers = []
    min_headers = []
    max_headers = []

    for h in headers:
        std_headers += [h + '_std']
        min_headers += [h + '_min']
        max_headers += [h + '_max']

    avg_env_df = pd.DataFrame()

    for u in new_env_df['날짜'].unique():
        a = new_env_df[new_env_df['날짜'] == u].mean().to_frame().T
        s = new_env_df[new_env_df['날짜'] == u].std().to_frame().T
        s.columns = std_headers
        min_ = new_env_df[new_env_df['날짜'] == u].min().to_frame().T
        min_.columns = min_headers
        max_ = new_env_df[new_env_df['날짜'] == u].max().to_frame().T
        max_.columns = max_headers
        
        hd = HD_cumulative_time(df=new_env_df[new_env_df['날짜'] == u])
        tt = temperature_cumulative_time(df=new_env_df[new_env_df['날짜'] == u])
        ta = temperature_average(df=new_env_df[new_env_df['날짜'] == u])
        su = suitable_info(df=new_env_df[new_env_df['날짜'] == u])
        
        stats = pd.concat([a, s, min_, max_, hd, tt, ta, su], axis=1)
        avg_env_df = pd.concat([avg_env_df, stats])

    avg_env_df = avg_env_df.reset_index()
    avg_env_df = avg_env_df.drop(columns=['index'])
    avg_env_df = avg_env_df.drop(columns=['날짜_std', '날짜_min', '날짜_max', '시간', '시간_std', '시간_min', '시간_max'])
    avg_env_df.to_excel(save_name, na_rep=0, header=True, index=False)

def growth_data_preprocessing(args):
    scr = args.growth.src_path
    files = args.growth.names
    save_name = os.path.join(args.growth.dest_path, args.growth.save_name)

    file_path = os.path.join(scr, files)
    df = pd.ExcelFile(file_path)
    growth_df = pd.read_excel(df, 'Sheet1')
    
    growth_df['leaf_area'] = growth_df['잎길이(cm)'] * growth_df['잎폭(cm)'] * growth_df['잎수(개)'] * (args.growth.planted_hills / args.growth.farm_area) * 0.6 / 10000

    growth_df['suitable_growth_length'] = 0
    growth_df.loc[growth_df['주간생육길이(cm)'] < 15, 'suitable_growth_length'] = -1
    growth_df.loc[growth_df['주간생육길이(cm)'] > 20, 'suitable_growth_length'] = 1

    growth_df['suitable_thickness'] = 0
    growth_df.loc[growth_df['줄기굵기(mm)'] < 9, 'suitable_thickness'] = -1
    growth_df.loc[growth_df['줄기굵기(mm)'] > 11, 'suitable_thickness'] = 1

    growth_df['suitable_leaf_length'] = 0
    growth_df.loc[growth_df['잎길이(cm)'] < 25, 'suitable_leaf_length'] = -1
    growth_df.loc[growth_df['잎길이(cm)'] > 35, 'suitable_leaf_length'] = 1

    growth_df['suitable_leaf_width'] = 0
    growth_df.loc[growth_df['잎폭(cm)'] < 20, 'suitable_leaf_width'] = -1
    growth_df.loc[growth_df['잎폭(cm)'] > 30, 'suitable_leaf_width'] = 1

    growth_df['suitable_number_of_leaf'] = 0
    growth_df.loc[growth_df['잎수(개)'] < 10, 'suitable_number_of_leaf'] = -1
    growth_df.loc[growth_df['잎수(개)'] > 15, 'suitable_number_of_leaf'] = 1

    growth_df['suitable_leaf_area'] = 0
    growth_df.loc[growth_df['leaf_area'] < 3, 'suitable_leaf_area'] = -1
    growth_df.loc[growth_df['leaf_area'] > 3.5, 'suitable_leaf_area'] = 1

    growth_df['suitable_flower_room'] = 0
    growth_df.loc[growth_df['개화화방위치(cm)'] < 10, 'suitable_flower_room'] = -1
    growth_df.loc[growth_df['개화화방위치(cm)'] > 15, 'suitable_flower_room'] = 1

    growth_df['suitable_flower_distance'] = 0
    growth_df.loc[growth_df['꽃과 줄기거리(cm)'] < 3, 'suitable_flower_distance'] = -1
    growth_df.loc[growth_df['꽃과 줄기거리(cm)'] > 4, 'suitable_flower_distance'] = 1

    growth_df['growth_type_score'] = growth_df['suitable_growth_length'] + growth_df['suitable_thickness'] + \
                                    growth_df['suitable_leaf_length'] + growth_df['suitable_leaf_width'] + \
                                    growth_df['suitable_number_of_leaf'] + growth_df['suitable_leaf_area'] + \
                                    growth_df['suitable_flower_room'] + growth_df['suitable_flower_distance']
    growth_df['growth_type'] = 0
    growth_df.loc[growth_df['growth_type_score'] < 0, 'growth_type'] = -1
    growth_df.loc[growth_df['growth_type_score'] > 0, 'growth_type'] = 1

    growth_df.to_excel(save_name, na_rep=0, header=True, index=False)

if __name__ == "__main__":
    args = ConfLoader(conf.config_path).opt

    env_data_preprocessing(args)

    growth_data_preprocessing(args)