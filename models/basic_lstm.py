import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import plot_model


'''
You can get a basic lstm model by a function call.
'''
def get_basic_lstm_model(input_shape):

    # debug - tuple
    assert len(input_shape) == 3, "Wrong input shape"
    print(input_shape)

    # build an LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    
    return model


def save_bl_losses(train_loss, val_loss):
    
    exp_dir = 'experiments/rda_basic_lstm'
    f1 = open(exp_dir+'/train_loss.txt', 'w')
    f2 = open(exp_dir+'/val_loss.txt', 'w')
    
    f1.write("train loss\n")
    f1.write("----------\n")
    for tl in train_loss:
        f1.write(str(tl)+"\n")
    f1.close()

    f2.write("train loss\n")
    f2.write("----------\n")
    for val in val_loss:
        f2.write(str(vl)+"\n")
    f2.close()


import pickle
class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


def save_bl_history(history):
    with open(savemodel_path+'/history', 'wb') as file:
        model_history= History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)


'''
You can get a basic lstm model by initiating a class instance as well.
todo: how to send inputs?
'''
class Basic_LSTM(keras.Model):
    def __init__(self, config):
        super(Basic_LSTM, self).__init__()
        self.input_shape_1 = config[1]
        self.input_shape_2 = config[2]
        #self.layer_1 = LSTM(units=50, return_sequences=True, input_shape=(self.input_shape_1, self.input_shape_2))
        self.layer_1 = LSTM(units=50, return_sequences=True)
        self.layer_2 = Dropout(0.3)
        self.layer_3 = LSTM(units=50, return_sequences=True)
        self.layer_4 = Dropout(0.3)
        self.layer_5 = LSTM(units=50, return_sequences=False)
        self.layer_6 = Dropout(0.3)
        self.layer_7 = Activation('relu')
        self.layer_8 = Dense(units=1)

    def call(self, inputs, training=False):
        #x = self.layer_1(inputs)
        self.inputs = inputs
        x = self.layer_1(input_shape=(self.inputs))
        if training:
            x = self.layer_2(x, training=training)
        x = self.layer_3(x)
        if training:
            x = self.layer_4(x, training=training)
        x = self.layer_5(x)
        if training:
            x = self.layer_6(x, training=training)
        outputs = self.layer_7(x)
        model = Model(inputs=inputs, outputs=outputs, name='rda_basic_lstm')
        return model
    
    def get_feature_importance(self):
        pass