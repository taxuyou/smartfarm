import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def get_rf_model(args,config=None):
    model = RandomForestRegressor()
    return model

