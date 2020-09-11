import tensorflow as tf
from tensorflow import keras


class MLP(tf.keras.Model):

    def __init__(self , input_shape, output_shape):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(output_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
