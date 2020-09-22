import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def get_mlp_model(args,config=None):
    # from tensorflow.keras import layers
    # inputs = config[0]
    # outputs = config[1]
    # model = keras.Sequential([
    #     layers.Dense(5,  activation='sigmoid', input_shape=[inputs]),
    #     # layers.Dense(10, activation='sigmoid'),
    #     layers.Dense(outputs)
    # ])
    model = RandomForestRegressor()
    return model

