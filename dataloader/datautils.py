import numpy as np
import pandas as pd
import os

__all__ = ['df2numpy']

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