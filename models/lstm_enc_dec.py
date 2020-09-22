import numpy as np
import tensorflow as tf
from tensorflow import keras

from .attention import Attention

class Encoder_Decoder(keras.Model):
    def __init__(self, input_shapes, output_shape, permute=True):
        super(Encoder_Decoder, self).__init__()
        # lstm encoder
        self.att1 = Attention(input_shapes[0], permute)
        self.att2 = Attention(input_shapes[1], permute)
        self.att3 = Attention(input_shapes[2], permute)
        self.att4 = Attention(input_shapes[3], permute)

        self.lstm1 = keras.layers.LSTM(output_shape[1])
        self.lstm2 = keras.layers.LSTM(output_shape[1])
        self.lstm3 = keras.layers.LSTM(output_shape[1])
        self.lstm4 = keras.layers.LSTM(output_shape[1])

        # decoder
        self.final_lstm = keras.layers.LSTM(output_shape[2], return_sequences=True)
        self.conv2d = keras.layers.Conv2D(output_shape[3], 3, activation='relu', padding="same", input_shape=output_shape[1:2])
        # self.inputs = tf.constant(input_shapes)
        # self.outputs = tf.constant(output_shape)

    def call(self, x, training=True):
        if training:
            tf.keras.backend.set_learning_phase(1) # tf.keras.backend.set_learning_phase will removed at 2020-10-11
        else:
            tf.keras.backend.set_learning_phase(0)
        x1, x2, x3, x4 = x

        out1 = self.att1(x1)
        out2 = self.att2(x2)
        out3 = self.att3(x3)
        out4 = self.att4(x4)

        out1 = self.lstm1(out1)
        out2 = self.lstm2(out2)
        out3 = self.lstm3(out3)
        out4 = self.lstm4(out4)

        shape1 = out1.shape
        out1 = tf.reshape(out1, [-1, shape1[1], 1])
        shape2 = out2.shape
        out2 = tf.reshape(out2, [-1, shape2[1], 1])
        shape3 = out3.shape
        out3 = tf.reshape(out3, [-1, shape3[1], 1])
        shape4 = out4.shape
        out4 = tf.reshape(out4, [-1, shape4[1], 1])

        out = tf.concat([out1, out2, out3, out4], axis=2)
        out = self.final_lstm(out)
        out_shape = out.shape
        out = tf.reshape(out, [-1, out_shape[1], out_shape[2], 1])
        
        out = self.conv2d(out)

        return out
    
    def get_attention_vector(self, x):
        x1, x2, x3, x4 = x

        out1 = self.att1.get_attention_vector(x1)
        out2 = self.att2.get_attention_vector(x2)
        out3 = self.att3.get_attention_vector(x3)
        out4 = self.att4.get_attention_vector(x4)

        return [out1, out2, out3, out4]

    def explain(self, dataset, return_heatmap=True):
        env_att = []
        growth_att = []
        product1_att = []
        product2_att = []

        for data, targets in dataset:
            att1, att2, att3, att4 = self.get_attention_vector(data)
            
            if return_heatmap:
                dols1 = np.array(att1).squeeze()
                dols2 = np.array(att2).squeeze()
                dols3 = np.array(att3).squeeze()
                dols4 = np.array(att4).squeeze()
            else:
                dols1 = np.mean(att1, axis=2).squeeze()
                dols2 = np.mean(att2, axis=2).squeeze()
                dols3 = np.mean(att3, axis=2).squeeze()
                dols4 = np.mean(att4, axis=2).squeeze()

            env_att.append(dols1)
            growth_att.append(dols2)
            product1_att.append(dols3)
            product2_att.append(dols4)
        
        env_att_final = np.mean(np.array(env_att), axis=0)
        growth_att_final = np.mean(np.array(growth_att), axis=0)
        product1_att_final = np.mean(np.array(product1_att), axis=0)
        product2_att_final = np.mean(np.array(product2_att), axis=0)

        return env_att_final, growth_att_final, product1_att_final, product2_att_final

    # def get_encoder_by_idx(self, index=0):
    #     if index == 0:
    #         return self.att1, self.lstm1
    #     elif index == 1:
    #         return self.att2, self.lstm2
    #     elif index == 2:
    #         return self.att3, self.lstm3
    #     elif index == 3:
    #         return self.att4, self.lstm4
    #     else:
    #         raise ValueError(index)

class Encoder_Decoder_Env(keras.Model):
    def __init__(self, input_shapes, output_shape, permute=True):
        super(Encoder_Decoder_Env, self).__init__()
        # lstm encoder
        self.att1 = Attention(input_shapes[0], permute)

        self.lstm1 = keras.layers.LSTM(output_shape[1])

        # decoder
        self.final_lstm = keras.layers.LSTM(output_shape[2], return_sequences=True)
        self.conv2d = keras.layers.Conv2D(output_shape[3], 3, activation='relu', padding="same", input_shape=output_shape[1:2])

    def call(self, x):
        x1 = x[0]
        out = self.att1(x1)

        out = self.lstm1(out)
        
        shape = out.shape
        out = tf.reshape(out, [shape[0], shape[1], 1])

        out = self.final_lstm(out)
        out_shape = out.shape
        out = tf.reshape(out, [out_shape[0], out_shape[1], out_shape[2], 1])
        
        out = self.conv2d(out)

        return out
    
    def get_attention_vector(self, x):
        x1 = x[0]
        out = self.att1.get_attention_vector(x1)

        return out

    def explain(self, dataset, return_heatmap=True):
        env_att = []

        for data, targets in dataset:
            att = self.get_attention_vector(data)
            
            if return_heatmap:
                dols = np.array(att).squeeze()
            else:
                dols = np.mean(att, axis=2).squeeze()

            env_att.append(dols)
        
        env_att_final = np.mean(np.array(env_att), axis=0)

        return env_att_final