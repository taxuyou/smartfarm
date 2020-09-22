import os

import pandas as pd
import numpy as np
import tensorflow as tf

from .datautils import df2numpy

__all__ = ['dataloader4lstm_enc_dec']

def dataloader4lstm_enc_dec(args):
    train_data_list = []
    train_names = args.data.train_data.names
    for name in train_names:
        path = args.data.train_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')

        if '초장(cm)' in sheet_df.columns:
            data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜'])[:args.data.num_data]
        elif '샘플번호' in sheet_df.columns:
            data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜', '샘플번호'])[:args.data.num_data]
        else:
            data = df2numpy(sheet_df, args.data.seek_days, offset=7, dropkey=['날짜'])[:args.data.num_data]
        
        tf_data = tf.convert_to_tensor(data, dtype=np.float32)
        train_data_list.append(tf_data)

    label_data_list = []
    label_names = args.data.label_data.names
    for name in label_names:
        path = args.data.label_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜', '샘플번호'])[:args.data.num_data]

        tf_data = tf.convert_to_tensor(data, dtype=np.float32)
        label_data_list.append(data)
    
    ds = []
    if args.model.config.avg:
        avg_label_data_list = tf.math.divide_no_nan(label_data_list[1], label_data_list[0])

        env, growth, pro1, pro2 = train_data_list

        for e, g, p1, p2, a in zip(env, growth, pro1, pro2, avg_label_data_list):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]]), tf.reshape(g, [1, g.shape[0], g.shape[1]]),
                tf.reshape(p1, [1, p1.shape[0], p1.shape[1]]), tf.reshape(p2, [1, p2.shape[0], p2.shape[1]])],
                tf.reshape(a, [1, a.shape[0], a.shape[1], 1])
            ])
    else:
        env, growth, pro1, pro2 = train_data_list
        pro3, pro4 = label_data_list

        for e, g, p1, p2, p3, p4 in zip(env, growth, pro1, pro2, pro3, pro4):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]]), tf.reshape(g, [1, g.shape[0], g.shape[1]]), 
                tf.reshape(p1, [1, p1.shape[0], p1.shape[1]]), tf.reshape(p2, [1, p2.shape[0], p2.shape[1]])], 
                tf.concat([tf.reshape(p3, [1, p3.shape[0], p3.shape[1], 1]), tf.reshape(p4, [1, p4.shape[0], p4.shape[1], 1])], axis=3)
            ])

    return ds

def dataloader4lstm_enc_dec_env(args):
    train_data_list = []
    train_names = args.data.train_data.names[0]
    path = args.data.train_data.path
    file_path = os.path.join(path, train_names)
    df = pd.ExcelFile(file_path)
    sheet_df = pd.read_excel(df, 'Sheet1')

    data = df2numpy(sheet_df, args.data.seek_days, offset=7, dropkey=['날짜'])[:args.data.num_data]
    
    tf_data = tf.convert_to_tensor(data, dtype=np.float32)
    train_data_list.append(tf_data)

    label_data_list = []
    label_names = args.data.label_data.names
    for name in label_names:
        path = args.data.label_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜', '샘플번호'])[:args.data.num_data]

        tf_data = tf.convert_to_tensor(data, dtype=np.float32)
        label_data_list.append(data)
    
    ds = []
    if args.model.config.avg:
        avg_label_data_list = tf.math.divide_no_nan(label_data_list[1], label_data_list[0])

        env = train_data_list[0]

        for e, a in zip(env, avg_label_data_list):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]])],
                tf.reshape(a, [1, a.shape[0], a.shape[1], 1])
            ])
    else:
        env = train_data_list
        pro3, pro4 = label_data_list

        for e, p3, p4 in zip(env, pro3, pro4):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]])], 
                tf.concat([tf.reshape(p3, [1, p3.shape[0], p3.shape[1], 1]), tf.reshape(p4, [1, p4.shape[0], p4.shape[1], 1])], axis=3)
            ])

    return ds

def make_data_frame(file_names, path, label=False, seek_days=43):

    data = dict()
    for name in file_names:
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        
        if "env" in name:
            data["env"] = sheet_df
        elif "growth" in name:
            data["growth"] = sheet_df
        elif "product_1" in name:
            data["product_1"] = sheet_df
        elif "product_2" in name:
            data["product_2"] = sheet_df
        elif "product_3" in name:
            data["product_3"] = sheet_df
        elif "product_4" in name:
            data["product_4"] = sheet_df
        else:
            raise ValueError(name)
        
    if not label:
        standard_index = 0
        date = data["product_1"].iloc[standard_index]["날짜"]
        env_index = data["env"].index[data["env"]["날짜"]==date].tolist()[0]
        num_samples = len(data["product_1"]) // len(data["product_1"]["날짜"].unique())

        while env_index < seek_days:
            standard_index += num_samples
            date = data["product_1"].iloc[standard_index]["날짜"]
            env_index = data["env"].index[data["env"]["날짜"]==date].tolist()[0]
        
        new_env_df = data["env"].iloc[index - seek_days + 1:]
        g_index = data["growth"].index[data["growth"]["날짜"]==date].tolist()[0]
        new_growth_df = data["growth"].iloc[g_index:]
        new_product_1_df = data["product_1"].iloc[standard_index:]
        new_product_2_df = data["product_2"].iloc[standard_index:]

        df_list = [new_env_df, new_growth_df, new_product_1_df, new_product_2_df]

    else:
        df_list = [data["product_3"], data["product_4"]]

    return df_list