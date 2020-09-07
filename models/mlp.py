import tensorflow as tf
from tensorflow import keras


class Mlp(tf.keras.Model):

    def __init__(self):
        super(Mlp, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(6)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
