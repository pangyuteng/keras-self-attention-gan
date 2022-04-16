# https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np

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
"""
class SelfAttention2D(keras.layers.Layer):
    def __init__(self,channel,trainable=True,**kwargs):
        super(SelfAttention2D, self).__init__(**kwargs)
        self.channel = channel # channel from prior layer
        self.trainable = trainable

    def build(self, input_shape):
        
        sn = tfa.layers.SpectralNormalization
        mykwargs = dict(kernel_size=1, strides=1, use_bias=False, padding="same",trainable=self.trainable)
        
        self.conv_f = sn(layers.Conv2D(self.channel // 8, **mykwargs))# [bs, h, w, c']
        self.conv_g = sn(layers.Conv2D(self.channel // 8, **mykwargs)) # [bs, h, w, c']
        self.conv_h = sn(layers.Conv2D(self.channel, **mykwargs)) # [bs, h, w, c]
        self.conv_v = sn(layers.Conv2D(self.channel, **mykwargs))# [bs, h, w, c]

        self.gamma = self.add_weight(name='gamma',shape=(1,),initializer='zeros',trainable=self.trainable)

    @staticmethod
    def hw_flatten(x) :
        shape = x.get_shape().as_list()
        dim = np.prod(shape[1:-1])
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

        return y



class SAGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()
        
        self.generator_attn = SelfAttention2D(32)

        model.add(Dense(196, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
 
        model.add(Dense(196))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        myshape = (14,14,1)
        model.add(Reshape(myshape))
        
        model.add(Conv2D(32,3,padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(self.generator_attn)

        model.add(Conv2DTranspose(16,3,strides=(2,2),padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1,(3,3),padding="same",activation='tanh'))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        
        self.discriminator_attn = SelfAttention2D(16)

        model = Sequential()
        model.add(Conv2D(16,(3,3),padding="same",input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(self.discriminator_attn)

        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1)) + 0.05*np.random.normal(0,1, (batch_size,1))
        fake = np.zeros((batch_size, 1)) + 0.05*np.random.normal(0,1, (batch_size,1))
        
        valid = valid.clip(0,1)
        fake = fake.clip(0,1)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            
            g_gamma = self.generator_attn.gamma.numpy()
            d_gamma = self.discriminator_attn.gamma.numpy()

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%% gamma: %1.5f] [G loss: %f, gamma: %1.5f] " % (epoch, d_loss[0], 100*d_loss[1], d_gamma, g_loss, g_gamma))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = SAGAN()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)