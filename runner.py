import os, sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import pandas as pd

import numpy as np
from numpy import asarray, concatenate

import xgboost as xgb
import pickle


import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from utils.radar import ComplexRadar
import json

from models import *
from utils import *
from dataloader import *
from optimizer import *

__all__ = ['Runner']




class Runner():
    def __init__(self, args):
        self.args = args

    def train(self):
        # ihshin
        if self.args.predict == 'multi_encoder_train':
            multi_encoder_train(self.args)
        # yhmoon
        elif self.args.predict == 'rda_tomato_train' and self.args.model.name == 'decision_tree':       
            decision_tree_train(self.args)
        elif self.args.predict == 'rda_tomato_train' and self.args.model.name == 'basic_lstm':  
            basic_lstm_train(self.args)
        elif self.args.predict == '2nd_tomato_train' and self.args.model.name == 'multiout_decision_tree':       
            multiout_decision_tree_train(self.args)
        elif self.args.predict == 'mlp_train':
            mlp_train(self.args)
        else:
            raise ValueError("Wrong Predict")
    
    def infer(self):
        # ihshin
        if self.args.predict == 'multi_encoder_inter':
            multi_encoder_inter(self.args)
        # yhmoon
        elif self.args.predict == 'rda_tomato_infer' and self.args.model.name == 'decision_tree':       
            decision_tree_infer(args)
        elif self.args.predict == 'rda_tomato_infer' and self.args.model.name == 'basic_lstm':  
            basic_lstm_infer(self.args)
        elif self.args.predict == '2nd_tomato_infer' and self.args.model.name == 'multiout_decision_tree':       
            multiout_decision_tree_infer(self.args)
        # jypark
        elif self.args.predict == 'mlp_train':
            mlp_infer(self.args)
        # ...
        else:
            raise ValueError("Wrong Predict")

    def run(self):
        if 'train' in self.args.predict:
            self.train()
        elif 'infer' in self.args.predict:
            self.infer()
        else:
            raise ValueError("Wrong Predict")

def build_model(inputs, outputs):
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Dense(256,  activation='relu', input_shape=[inputs]),
        layers.Dense(256, activation='relu'),
        layers.Dense(outputs)
    ])
    optimizer = tf.keras.optimizers.SGD(0.01)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def mlp_train(args):
    print('mlp_train')
    ds = get_dataloader(args)

    # First Dataset Training
    train_dataset = ds['first_dataset'].sample(frac=0.8,random_state=0)
    test_dataset  = ds['first_dataset'].drop(train_dataset.index)

    train_labels = train_dataset[['주간생육길이(cm)','d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','dleaf_area']].copy()
    test_labels = test_dataset[['주간생육길이(cm)','d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','dleaf_area']].copy()

    train_dataset.drop(['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area'],axis=1,inplace=True)
    test_dataset.drop(['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area'],axis=1,inplace=True)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()


    print(train_dataset.shape)
    print(test_dataset.shape)

    normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
    normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']
    model = build_model(len(train_dataset.keys()), len(train_labels.keys()))
    model.summary()

    history = model.fit(
    normed_train_data.values, train_labels,
    epochs=args.train.epochs,  validation_split = 0.2, verbose=0,callbacks=[tfdocs.modeling.EpochDots()])

    exp_dir = args.util.path
    filename = args.util.first_model
    model.save(exp_dir + filename)
    train_stats.to_csv(args.util.path+args.util.train_stats)  

    test_dataset.to_csv(exp_dir+"/test_dataset.csv")
    test_labels.to_csv(exp_dir+"/test_labels.csv")

    mlp_infer(args)
    return model

def clip(input,ranges):
    for col in range(len(input.columns)):
        for row in range(len(input)):
            if input.iloc[row,col] <= ranges[col][0]:
                input.iloc[row,col] = ranges[col][0]
            elif input.iloc[row,col] >= ranges[col][1]:
                input.iloc[row,col] = ranges[col][1]
            else:
                input.iloc[row,col]
    return input

def mlp_infer(args):
    print('mlp_infer')
    ds = get_dataloader(args)
    exp_dir = args.util.path
    filename = args.util.model
    train_stats = pd.read_csv(exp_dir+args.util.train_stats,index_col=0)

    test_dataset = pd.read_csv(exp_dir+"/test_dataset.csv",index_col=0)

    normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']
    normed_test_data = normed_test_data.dropna()
    
    test_labels = pd.read_csv(exp_dir + "/test_labels.csv",index_col=0)

    model = build_model(len(test_dataset.keys()),len(test_labels.keys()))
    model.load_weights(exp_dir+filename)

    ypred = pd.DataFrame(model.predict(normed_test_data))
    
    variables = ('Grown_height(cm)', 'Thickness(mm)', 'Leaf Length(cm)', 'Leaf Width(cm)', 'Leaf Area')

    max_data =(20, 11, 35, 30, 3.5) 
    min_data =(15, 9, 25, 20, 3)

    ranges = [(5, 30), (5, 15), (5, 55), (0, 50), (2.0, 4.5)]         
    

    for x in range(len(ypred)):
        ypred= clip(ypred,ranges)
        test_labels = clip(test_labels,ranges)

        fig1 = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig1, variables, ranges)
        radar.fill(max_data,'g')
        radar.fill(min_data,color='w')
        radar.plot(ypred.iloc[x],'r')
        radar.plot(test_labels.iloc[x],'b')
        
        plt.savefig(exp_dir+"/growth_images/pred"+str(x)+".png")



def multi_encoder_train(args):
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
                harvest_path = os.path.join(save_path, 'harvest.png')
                draw_harvest_per_sample(out, targets, harvest_path)
        
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

def multi_encoder_inter(args):
    # get data loader - [[data], [label]]
    ds = get_dataloader(args)
    # input parameters for lstm_inc_dec model
    args.model.config.input_shapes = get_input_shapes(args)
    args.model.config.output_shape = get_output_shapes(args)

    save_path = args.util.save_path
    pretrained_path = args.model.pretrained_path
    
    # Select device
    with tf.device(args.device.name):
        # model create
        model = get_model(args)
        # optimizer create
        optimizer = get_optimizer(args)
        # load weights from file
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint.restore(tf.train.latest_checkpoint(pretrained_path)).expect_partial()

        for data, targets in ds:
            out = model(data, training=False)
            output_path = os.path.join(save_path, 'output.csv')
            groundtruth_path = os.path.join(save_path, 'groundtruth.csv')
            tensor2csv(output_path, out)
            tensor2csv(groundtruth_path, targets)
            harvest_path = os.path.join(save_path, 'harvest.png')
            draw_harvest_per_sample(out, targets, harvest_path)

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
