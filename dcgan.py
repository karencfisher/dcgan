import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import LeakyReLU, Flatten, Input
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


class DCGAN:
    def __init__(self, lr=2e-4, model_path=None):
        self.optimizer = Adam(learning_rate=lr)
        self.loss = BinaryCrossentropy()

        if model_path is None:
            self.generator = self.__generator()
            self.discriminator = self.__discriminator()
            self.dcgan = self.__dcgan()
        else:
            disc_file = os.path.join(model_path, 'discrminator.h5')
            self.discriminator = load_model(disc_file)

            gen_file = os.path.join(model_path, 'generator.h5')
            self.generator = load_model(gen_file)

            dcgan_file = os.path.join(model_path, 'dcgan.h5')
            self.dcgan = load_model(dcgan_file)

    def __generator(self):
        generator = Sequential([
            Conv2DTranspose(1024, (4, 4), input_shape=(1, 1, 100)),
            LeakyReLU(0.2),

            Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', 
                            activation='tanh')
        ])

        generator.compile(optimizer=self.optimizer, loss=self.loss)
        return generator

    def __discriminator(self):
        discriminator = Sequential([
            Conv2D(128, (4, 4), strides=(2, 2), padding='same', 
                   input_shape=(64, 64, 1)),
            LeakyReLU(0.2),

            Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Conv2D(1024, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),

            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        discriminator.compile(optimizer=self.optimizer, loss=self.loss)
        return discriminator

    def __dcgan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(1, 1, 100))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        dcgan = Model(gan_input, gan_output)

        dcgan.compile(optimizer=self.optimizer, loss=self.loss)
        return dcgan
    
    def train(self, X, num_epochs, batch_size, verbose=5):
        d_losses = []
        g_losses = []
        images = tf.image.resize(X, (64, 64))       

        for epoch in range(num_epochs):
            batch_count = int(X.shape[0] / batch_size)
            start = 0

            for _ in tqdm(range(batch_count), ascii=True, desc=f'Epoch {epoch+1}'):
                stop = start + batch_size
                real_imgs = images[start: stop]
            
                noise = np.random.normal(0, 1, size=(batch_size, 1, 1, 100))
                generated_imgs = self.generator.predict(noise, verbose=0)
                imgs = np.concatenate([real_imgs, generated_imgs])
            
                real_y = np.ones((batch_size, 1)) * 0.9
                fake_y = np.zeros((batch_size, 1))
                labels = np.concatenate([real_y, fake_y])

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(imgs, labels)

                noise = np.random.normal(0, 1, size=(batch_size, 1, 1, 100))
                real_y = np.ones((batch_size, 1))

                self.discriminator.trainable = False
                g_loss = self.dcgan.train_on_batch(noise, real_y)

                start += batch_size
        
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if verbose and (epoch == 0 or(epoch + 1) % verbose == 0):
                self.generate(10, epoch+1, display=True)
        
        return d_losses, g_losses

    def generate(self, n_examples, epoch=None, display=False):
        noise = np.random.normal(0, 1, size=(n_examples, 1, 1, 100))
        gen_imgs = self.generator.predict(noise, verbose=0)

        if display:
            rows = n_examples // 5
            rows += 1 if n_examples % 5 > 0 else 0
            fig, ax = plt.subplots(rows, 5, figsize=(5, 3))
            fig.patch.set_facecolor('white')
            for indx in range(n_examples):
                img = gen_imgs[indx].reshape(64, 64)
                i, j = indx // 5, indx % 5
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

            label = 'Epoch {0}'.format(epoch)
            fig.text(0.5, 0.04, label, ha='center')
            plt.show()
            
        return gen_imgs

    def save_model(self, model_path):
        disc_file = os.path.join(model_path, 'discrminator.h5')
        self.discriminator.save(disc_file)

        gen_file = os.path.join(model_path, 'generator.h5')
        self.generator.save(gen_file)

        dcgan_file = os.path.join(model_path, 'dcgan.h5')
        self.dcgan.save(dcgan_file)
