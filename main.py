import os, sys
import errno, argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import pandas as pd

import numpy as np
from numpy import asarray, concatenate

import xgboost as xgb
import pickle

from models import *
from utils import *
from dataloader import *
from optimizer import *


parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, default='./config.json')
conf = parser.parse_args()


def check_env():
    # check deep learning platform
    print("TensorFlow: %s" %(tf.__version__))
    print("Keras: %s" %(keras.__version__))
 
    # check available gpu devices
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())


def check_dir():
    exp_dir = ['./experiments', './experiments/rda_decision_tree', './experiments/rda_basic_lstm']
    def check(dirs):
        for i in range(len(dirs)):
            try:
                os.makedirs(dirs[i])
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
    check(exp_dir)


#@tf.function
def train_step(inp, targ, model, optimizer):

    with tf.GradientTape() as tape:
        out = model(inp, training=True)
        loss = keras.losses.mean_squared_error(targ, out)
    gradients =tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, out

def inference(inp, model):
    out = model(inp, training=False)
    return out

def mlp_predict(args):
    ds = get_dataloader(args)


def oneshot_predict(args):
    
    # get data loader - [[data], [label]]
    ds = get_dataloader(args)
    # input parameters for lstm_inc_dec model
    args.model.config.input_shapes = get_input_shapes(args)
    args.model.config.output_shape = get_output_shapes(args)

    save_path = args.util.save_path
    
    # Select device
    with tf.device(args.device.name):
        # model create
        model = get_model(args)
        # optimizer create
        optimizer = get_optimizer(args)
        # loss metric setting
        train_loss = keras.metrics.Mean(name="train_loss")
        # model save setting
        checkpoint_dir = os.path.join(save_path, 'ckpt/ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # Set epochs
        EPOCHS = args.train.epochs

        # Start training
        for epoch in range(EPOCHS):
            train_loss.reset_states()

            for data, targets in ds:
                loss, out = train_step(data, targets, model, optimizer)

                train_loss(loss)
            
            if epoch % 10 == 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch+1, train_loss.result()))
            
            if epoch % 100 == 0 or epoch == (EPOCHS - 1):
                checkpoint.save(file_prefix=checkpoint_dir)
                out = inference(data, model)
                output_path = os.path.join(save_path, 'output.csv')
                groundtruth_path = os.path.join(save_path, 'groundtruth.csv')
                tensor2csv(output_path, out)
                tensor2csv(groundtruth_path, targets)
        
        # Drawing heatmap and bar chart for explanation
        if args.model.config.explain:
            if args.model.config.env_only:
                if args.util.env_heatmap.avail:
                    e = model.explain(ds, return_heatmap=True)
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, x_labels=x_labels, y_labels=y_labels, filename=heatmap_path)
            else:
                if args.util.env_heatmap.avail:
                    e, _, _, _ = model.explain(ds, return_heatmap=True)
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, x_labels=x_labels, y_labels=y_labels, filename=heatmap_path)
                
                if args.util.growth_heatmap.avail:
                    _, g, _, _ = model.explain(ds, return_heatmap=True)
                    x_labels = args.util.growth_heatmap.x_labels
                    heatmap_path = os.path.join(save_path, args.util.growth_heatmap.name)
                    draw_bargraph(data=g, filename=heatmap_path, x_labels=x_labels)



def decision_tree_train(args):
    print("xgboost version:", xgb.__version__)

    df, data, dataset = get_dataloader(args)
    if int(args.train.test_samples) == 0 and len(dataset) != 2:
        raise ValueError(args.model.name + '- dataset is missing')
    elif int(args.train.test_samples) > 0 and len(dataset) != 4:
        raise ValueError(args.model.name + '- dataset is missing')
    train_X = dataset[0]
    train_y = dataset[1]
    test_X = None 
    test_y = None
    if len(dataset) == 4:
        test_X = dataset[2]
        test_y = dataset[3]

    # model
    model = get_model(args)
    
    # train
    if args.model.pretrained:
        # load pretrained model
        pass
    model.fit(train_X, train_y)

    # save model to file using pickle
    exp_dir = './experiments/rda_decision_tree'
    pickle.dump(model, open(exp_dir+"/rda_dt_model.dat", "wb"))

    # test
    # construct multiple inputs for new predictions
    # for this, we use all samples (36)
    # so, overfitting is expected for sure
    rows = train_X.shape[0]
    if len(dataset) == 4:
        rows = test_X.shape[0]
    y_true = []
    y_pred = []
    for row_index in range(0, rows):
        row = data.values[row_index, :-1]
        y_true.append(data.values[row_index, -1])
        # make a one-step prediction
        y_pred.append(model.predict(asarray([row])))
        #print('Test Input: %s' %(row))
        print('Ground True: %.9f, Predicted: %.9f' %(y_true[row_index], y_pred[row_index]))  
    
    # calculate error
    error = get_error(args.train.metric, y_true, y_pred)
    print("%s: %.6f" %(str(args.train.metric), error))
    
    # save plot of expected vs. predicted
    draw_dt_result_compare_1(rows, y_true, y_pred)
    draw_dt_result_compare_2(y_true, y_pred)

    # save plot of feature importance
    draw_dt_feature_importance(model)
    
    # save top-n feature importance
    top = 10
    lookback = int(args.data.supervised_data.lookback)
    save_feature_importance(model, df, lookback, top)


def decision_tree_infer(args):

    # test samples for cross validation must be larger than 0 
    if int(args.infer.test_samples) == 0:
        raise ValueError(args.model.name + 'no test samples are assigned')

    # load model from file
    model = pickle.load(open("./experiments/rda_decision_tree/rda_dt_model.dat", "rb"))
   
    # load test dataset
    # todo: use get_dataloader_infer(args), there is no ground truth
    df, data, dataset = get_dataloader(args)
    if int(args.train.test_samples) == 0 and len(dataset) != 2:
        raise ValueError(args.model.name + '- dataset is missing')
    elif int(args.train.test_samples) > 0 and len(dataset) != 4:
        raise ValueError(args.model.name + '- dataset is missing')
    train_X = dataset[0]
    train_y = dataset[1]
    test_X = None 
    test_y = None
    if len(dataset) == 4:
        test_X = dataset[2]
        test_y = dataset[3]

    y_true = []
    y_pred = []    
    rows = 0
    if test_X.size > 0 and test_y.size > 0:
        rows = test_X.shape[0]
    
    # infer
    for row_index in range(0, rows):
        row = data.values[row_index, :-1]
        y_true.append(data.values[row_index, -1])
        # make a one-step prediction
        y_pred.append(model.predict(asarray([row])))
        #print('Test Input: %s' %(row))
        print('Ground True: %.9f, Predicted: %.9f' %(y_true[row_index], y_pred[row_index]))  
    
    # evaluate predictions - calculate error
    # todo: show predicted value (harvest yield)
    error = get_error(args.infer.metric, y_true, y_pred)
    print("%s: %.6f" %(str(args.infer.metric), error))


def basic_lstm_train(args):
    
    # test samples for cross validation must be larger than 0 
    if int(args.train.test_samples) == 0:
        raise ValueError(args.model.name + 'no test samples are assigned')

    df, data, dataset, scaler = get_dataloader(args)
    if len(dataset) != 4:
        raise ValueError(args.model.name + '- dataset is missing')
    
    # check
    history = tf.keras.callbacks.History()
    #history = History()

    # assign a gpu device to a training session
    with tf.device(args.device.name):
        
        # dataset
        train_X, train_y, test_X, test_y = dataset

        # model
        model = get_model(args, train_X.shape)

        # optimizer
        optimizer = get_optimizer(args)
        
        # loss
        train_loss = keras.metrics.Mean(name="train_loss")

        # checkpoint
        # https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
        checkpoint_path = './experiments/rda_basic_lstm/rda_basic_lstm_{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=50)
        
        # epochs
        epochs = args.train.epochs

        try:
            '''
            # Basic_LSTM
            lookback = int(args.data.supervised_data.lookback) # loockback timesteps
            input_dim = 35 # int(test_X.shape[2]) or features?
            inputs = keras.Input(shape=(lookback, input_dim), name="basic_lstm_input_layer") 
            out = model(inputs, training=True)
            out.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            out.summary()
            '''
            # get_basic_lstm_model
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            model.summary()

            # train and validation
            history = model.fit(train_X, train_y,
                                epochs=epochs, batch_size=args.train.batch, 
                                validation_data=(test_X, test_y),
                                callbacks=[cp_callback], 
                                verbose=2, shuffle=False)
        except (OSError, ValueError, RuntimeError, TypeError, NameError) as err:
            print("Error: {0}".format(err))
            sys.exit()
    
    # save history of train loss and validation loss
    # type: list
    print(type(history.history['loss']))

    if history != None:
        draw_bl_losses(history.history['loss'], history.history['val_loss'])
        #save_bl_history(history.history['loss'], history.history['val_loss'])
    else:
        raise ValueError("History shouldn't be None")
 
    # evaluation for new data
    y_pred = model.predict(test_X)

    # reshape test input 3D (samples, lookback timesteps, features) 
    # to 2D (samples, lookback timesteps * features)
    print(test_X.shape)
    test_X = test_X.reshape((test_X.shape[0], int(test_X.shape[1] * test_X.shape[2])))
    print(test_X.shape)

    # invert scaling for forecast
    features = int(df.shape[1])
    inv_y_pred = concatenate((y_pred, test_X[:, -features+1:]), axis=1)
    inv_y_pred = scaler.inverse_transform(inv_y_pred)
    inv_y_pred = inv_y_pred[:,-1]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y_true = concatenate((test_y, test_X[:, -features+1:]), axis=1)
    inv_y_true = scaler.inverse_transform(inv_y_true)
    inv_y_true = inv_y_true[:,-1]

    print("ground truth: {}".format(inv_y_true))
    print("predictions: {}".format(inv_y_pred))

    # calculate error
    error = get_error(args.train.metric, inv_y_true, inv_y_pred)
    print("%s: %.6f" %(str(args.train.metric), error))

    # plot expected vs preducted
    rows = test_X.shape[0]
    draw_bl_result_compare_1(rows, inv_y_true, inv_y_pred)

    # plot predictions vs. ground truth
    draw_bl_result_compare_2(inv_y_true, inv_y_pred)  


def basic_lstm_infer(args):
    pass


def multiout_decision_tree_train(args):
    pass


def multiout_decision_tree_train(args):   
    pass


if __name__ =='__main__':
    
    args = ConfLoader(conf.config_path).opt

    print("Smart Farm Analytics v1.0")

    check_env()

    check_dir()
    
    # ihshin
    if args.predict == 'oneshot_predict':
        oneshot_predict(args)
    # yhmoon
    elif args.predict == 'rda_tomato_train' and args.model.name == 'decision_tree':       
        decision_tree_train(args)
    elif args.predict == 'rda_tomato_infer' and args.model.name == 'decision_tree':       
        decision_tree_infer(args)
    elif args.predict == 'rda_tomato_train' and args.model.name == 'basic_lstm':  
        basic_lstm_train(args)
    elif args.predict == 'rda_tomato_infer' and args.model.name == 'basic_lstm':  
        basic_lstm_infer(args)
    elif args.predict == '2nd_tomato_train' and args.model.name == 'multiout_decision_tree':       
        multiout_decision_tree_train(args)
    elif args.predict == '2nd_tomato_infer' and args.model.name == 'multiout_decision_tree':       
        multiout_decision_tree_infer(args)
    # jypark
    # ...
    # ...
    else:
        raise ValueError("Wrong Predict")