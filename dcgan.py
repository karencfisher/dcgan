import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import LeakyReLU, Flatten, Input
from tensorflow.keras.layers import ReLU, BatchNormalization
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import initializers


class Preprocessor:
    '''
    Center the image data and divide by the nex maximum value
    e.g., if pixels in range [0, 255] subtract 127.5 to center
    and divide by 127.5, resulting in distribution -1 to 1
    '''
    def fit(self, X):
        self.median = np.max(X) / 2

    def transform(self, X):
        Z = (X.astype(np.float32) - self.median) / self.median
        return Z

    def inverse(self, Z):
        X = Z * self.median + self.median
        return X


class DCGAN:
    def __init__(self, input_shape, preprocessor=None, optimizer=None, 
                 lr=2e-4, latent_dim=100, model_path=None):
        if preprocessor is None:
            self.preprocessor = Preprocessor()
        else:
            self.preprocessor = preprocessor

        if optimizer is None:
            self.optimizer = Adam(learning_rate=lr, beta_1=0.5)
        else:
            self.optimizer = optimizer
            
        self.loss = BinaryCrossentropy()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.w_init = initializers.RandomNormal(mean=0, stddev=0.02)

        if model_path is None:
            self.generator = self.__generator(channels=input_shape[2])
            self.discriminator = self.__discriminator(channels=input_shape[2])
            self.dcgan = self.__dcgan()
        else:
            disc_file = os.path.join(model_path, 'discriminator.h5')
            self.discriminator = load_model(disc_file)

            gen_file = os.path.join(model_path, 'generator.h5')
            self.generator = load_model(gen_file)

            dcgan_file = os.path.join(model_path, 'dcgan.h5')
            self.dcgan = load_model(dcgan_file)

    def __generator(self, channels=1):
        generator = Sequential([
            Conv2DTranspose(1024, (4, 4), input_shape=(1, 1, self.latent_dim),
                            kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2DTranspose(channels, (4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=self.w_init, 
                            activation='tanh')
        ])

        generator.compile(optimizer=self.optimizer, loss=self.loss)
        return generator

    def __discriminator(self, channels=1):
        discriminator = Sequential([
            Conv2D(128, (4, 4), strides=(2, 2), padding='same', 
                   input_shape=(64, 64, channels),
                   kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2D(256, (4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2D(512, (4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Conv2D(1024, (4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=self.w_init),
            LeakyReLU(0.2),

            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        discriminator.compile(optimizer=self.optimizer, loss=self.loss)
        return discriminator

    def __dcgan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(1, 1, self.latent_dim))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        dcgan = Model(gan_input, gan_output)

        dcgan.compile(optimizer=self.optimizer, loss=self.loss)
        return dcgan
    
    def train(self, X, num_epochs, batch_size, begin=0, verbose=5, image_path=None, 
              checkpoint_path=None):
        d_losses = []
        g_losses = []
        images = tf.image.resize(X, (64, 64))
        start_time = time.time()      

        for epoch in range(num_epochs):
            batch_count = int(X.shape[0] / batch_size)
            start = 0

            for _ in tqdm(range(batch_count), ascii=True, desc=f'Epoch {begin+epoch+1}'):
                stop = start + batch_size
                real_imgs = images[start: stop]
            
                noise = np.random.normal(0, 1, size=(batch_size, 1, 1, self.latent_dim))
                generated_imgs = self.generator.predict(noise, verbose=0)
                imgs = np.concatenate([real_imgs, generated_imgs])
            
                real_y = np.ones((batch_size, 1)) * 0.9
                fake_y = np.zeros((batch_size, 1))
                labels = np.concatenate([real_y, fake_y])

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(imgs, labels)

                noise = np.random.normal(0, 1, size=(batch_size, 1, 1, self.latent_dim))
                real_y = np.ones((batch_size, 1))

                self.discriminator.trainable = False
                g_loss = self.dcgan.train_on_batch(noise, real_y)

                start += batch_size
        
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if checkpoint_path:
                self.save_model(checkpoint_path)

            if verbose and (epoch == 0 or(epoch + 1) % verbose == 0):
                self.generate(10, begin+epoch+1, display=True, image_folder=image_path)
        
        elapsed_time = time.time() - start_time
        hr = int(elapsed_time // 3600)
        elapsed_time %= 3600
        min = int(elapsed_time // 60)
        sec = int(elapsed_time % 60)
        print(f'Total training time: {hr}:{min:02d}:{sec:02d}')
        return d_losses, g_losses

    def generate(self, n_examples, epoch=None, display=False, image_folder=None):
        noise = np.random.normal(0, 1, size=(n_examples, 1, 1, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        gen_imgs = tf.image.resize(gen_imgs, self.input_shape[:2]).numpy()
        if display:
            self.display_samples(n_examples, gen_imgs, epoch, image_folder)
        return gen_imgs

    def display_samples(self, n_examples, gen_imgs, epoch, image_folder):
        rows = n_examples // 5
        rows += 1 if n_examples % 5 > 0 else 0
        fig, ax = plt.subplots(rows, 5, figsize=(5, 3))
        fig.patch.set_facecolor('white')
        for indx in range(n_examples):
            img = self.preprocessor.inverse(gen_imgs[indx])
            img = img.astype(int)
            i, j = indx // 5, indx % 5
            ax[i, j].imshow(img, cmap=plt.cm.binary)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        if image_folder is not None:
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)
            file_path = os.path.join(image_folder, f'epoch_{epoch}.jpg')
            plt.savefig(file_path)
        plt.show()

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            
        disc_file = os.path.join(model_path, 'discriminator.h5')
        self.discriminator.save(disc_file)

        gen_file = os.path.join(model_path, 'generator.h5')
        self.generator.save(gen_file)

        dcgan_file = os.path.join(model_path, 'dcgan.h5')
        self.dcgan.save(dcgan_file)
