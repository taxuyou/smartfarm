import numpy as np
import pandas as pd
import os

__all__ = ['df2numpy', 'get_input_shapes', 'get_output_shapes']

def df2numpy(df, timestep, offset=1, dropkey=None):
    # dropkey: list of key
    numData = len(df) - timestep + 1
    dataset = []
    for i in range(0, numData, offset):
        if dropkey != None:
            data = df[i:i+timestep].drop(columns=dropkey).values.tolist()
        else:
            data = df[i:i+timestep].values.tolist()
        dataset.append(data)
    
    return np.array(dataset)

def get_input_shapes(args):

    shape_list = []
    
    names = args.data.train_data.names
    
    for name in names:
        path = args.data.train_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        
        if '초장(cm)' in sheet_df.columns:
            data = df2numpy(sheet_df, args.data.num_samples, offset=1, dropkey=['주차'])[:args.data.num_data]
        elif '샘플번호' in sheet_df.columns:
            data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜', '샘플번호'])[:args.data.num_data]
        else:
            data = df2numpy(sheet_df, args.data.seek_days, offset=7, dropkey=['날짜'])[:args.data.num_data]

        shape_list.append(data.shape)

    return shape_list

def get_output_shapes(args):
    shape_list = []

    names = args.data.label_data.names
    for name in names:
        path = args.data.label_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        data = df2numpy(sheet_df, args.data.num_samples, offset=args.data.num_samples, dropkey=['날짜', '샘플번호'])[:args.data.num_data]
        
        shape_list.append(data.shape)
    
    if args.model.config.permute:
        output_shape = [shape_list[0][0], shape_list[0][1], shape_list[0][2], 2]
    else:
        output_shape = [shape_list[0][0], shape_list[0][1], shape_list[0][2], 1]

    return output_shape