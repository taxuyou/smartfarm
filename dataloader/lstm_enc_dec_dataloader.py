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
            data = df2numpy(sheet_df, 1, offset=args.data.num_samples, dropkey=['주차'])[:args.data.num_data]
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