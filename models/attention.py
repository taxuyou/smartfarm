import tensorflow as tf
from tensorflow import keras

class Attention(keras.Model):
    def __init__(self, input_shapes, permute=True):
        super(Attention, self).__init__()
        self.permute = permute
        if self.permute:
            self.permute = keras.layers.Permute((2,1))
            self.attention_permute = keras.layers.Permute((2,1), name='attention_vec')
            self.attention_weights = keras.layers.Dense(input_shapes[1], activation='softmax')
        else:
            self.attention_weights = keras.layers.Dense(input_shapes[2], activation='softmax')
        self.mul = keras.layers.Multiply()
    
    def call(self, x):
        if self.permute:
            a = self.permute(x)
            a = self.attention_weights(a)
            a_probs = self.attention_permute(a)
        else:
            a_probs = self.attention_weights(x)

        out = self.mul([x, a_probs])

        return out

    def get_attention_vector(self, x):
        if self.permute:
            a = self.permute(x)
            a = self.attention_weights(a)
            a_probs = self.attention_permute(a)
        else:
            a_probs = self.attention_weights(x)

        return a_probs