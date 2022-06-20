
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

"""
references
https://arxiv.org/pdf/1805.08318.pdf
https://github.com/brain-research/self-attention-gan
https://github.com/taki0112/Self-Attention-GAN-Tensorflow
https://lilianweng.github.io/posts/2018-06-24-attention
https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras
"""
class SelfAttention(keras.layers.Layer):
    def __init__(self,channel,trainable=True,
        convclass=layers.Conv2D,**kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.ConvClass = convclass
        self.channel = channel # channel from prior layer
        self.trainable = trainable

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        
        sn = tfa.layers.SpectralNormalization
        mykwargs = dict(kernel_size=1, strides=1, use_bias=False, padding="same",trainable=self.trainable)
        
        self.conv_f = sn(self.ConvClass(self.channel // 8, **mykwargs))# [bs, h, w, c']
        self.conv_g = sn(self.ConvClass(self.channel // 8, **mykwargs)) # [bs, h, w, c']
        self.conv_h = sn(self.ConvClass(self.channel, **mykwargs)) # [bs, h, w, c]
        self.conv_v = sn(self.ConvClass(self.channel, **mykwargs))# [bs, h, w, c]

        self.gamma = self.add_weight(name='gamma',shape=(1,),initializer='zeros',trainable=self.trainable)

    @staticmethod
    def hw_flatten(x) :
        shape = x.get_shape().as_list()
        dim = keras.backend.prod(shape[1:-1])
        return tf.reshape(x, [-1, dim, shape[-1]])

    def call(self, inputs):
        f = self.conv_f(inputs) # key
        g = self.conv_g(inputs) # query
        h = self.conv_h(inputs) # value

        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True) # [bs, N, N], N = h * w
        beta = tf.nn.softmax(s) # beta is your attention map
        
        o = tf.matmul(beta, self.hw_flatten(h)) # [bs, N, C]
        input_shape = (-1,)+tuple(inputs.get_shape().as_list()[1:])
        o = tf.reshape(o, shape=input_shape) # [bs, h, w, C]
        o = self.conv_v(o)

        y = self.gamma * o + inputs

        return y, beta
