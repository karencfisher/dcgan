import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D, Dropout
from tensorflow.keras.layers import LeakyReLU, Flatten, Input, Reshape
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy


class Preprocessor:
    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)

    def transform(self, X):
        Z = (X - self.mean) / self.std
        return Z

    def inverse(self, Z):
        X = Z * self.std + self.mean
        return X


class DCGAN:
    def __init__(self, channels=1, lr=4e-4, latent_dim=100, model_path=None):
        self.latent_dim = latent_dim
        self.optimizer = RMSprop(learning_rate=lr, clipvalue=1.0, decay=1e-8)
        self.loss = BinaryCrossentropy()
        self.preprocessor = Preprocessor()

        if model_path is None:
            self.generator = self.__generator(channels=channels)
            self.discriminator = self.__discriminator(channels=channels)
            self.dcgan = self.__dcgan()
        else:
            disc_file = os.path.join(model_path, 'discrminator.h5')
            self.discriminator = load_model(disc_file)

            gen_file = os.path.join(model_path, 'generator.h5')
            self.generator = load_model(gen_file)

            dcgan_file = os.path.join(model_path, 'dcgan.h5')
            self.dcgan = load_model(dcgan_file)

    def __generator(self, channels=1):
        generator = Sequential([
            Dense(128 * 16 * 16, input_shape=(self.latent_dim,)),
            LeakyReLU(),
            Reshape((16, 16, 128)),

            Conv2D(256, (5, 5), padding='same'),
            LeakyReLU(),

            Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(),

            Conv2D(256, (5, 5), padding='same'),
            LeakyReLU(),
            Conv2D(256, (5, 5), padding='same'),
            LeakyReLU(),

            Conv2D(channels, (7, 7), activation='tanh', padding='same')
        ])
        
        return generator

    def __discriminator(self, channels=1):
        discriminator = Sequential([
            Conv2D(128, (3, 3), input_shape=(32, 32, channels)),
            LeakyReLU(),

            Conv2D(128, (4, 4), strides=(2, 2)),
            LeakyReLU(),

            Conv2D(512, (4, 4), strides=(2, 2)),
            LeakyReLU(),

            Conv2D(1024, (4, 4), strides=(2, 2)),
            LeakyReLU(),

            Flatten(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        discriminator.compile(optimizer=self.optimizer, loss=self.loss)
        return discriminator

    def __dcgan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        dcgan = Model(gan_input, gan_output)

        dcgan.compile(optimizer=self.optimizer, loss=self.loss)
        return dcgan
    
    def train(self, X, num_epochs, batch_size, verbose=5, checkpoint_path=None):
        d_losses = []
        g_losses = []
        images = tf.image.resize(X, (32, 32))
        start_time = time.time()      

        for epoch in range(num_epochs):
            batch_count = int(X.shape[0] / batch_size)
            start = 0

            for _ in tqdm(range(batch_count), ascii=True, desc=f'Epoch {epoch+1}'):
                stop = start + batch_size
                real_imgs = images[start: stop]
            
                noise = np.random.normal(size=(batch_size, self.latent_dim))
                generated_imgs = self.generator.predict(noise, verbose=0)
                imgs = np.concatenate([real_imgs, generated_imgs])
            
                real_y = np.ones((batch_size, 1))
                fake_y = np.zeros((batch_size, 1))
                labels = np.concatenate([real_y, fake_y])
                labels += 0.05 * np.random.random(labels.shape)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(imgs, labels)

                noise = np.random.normal(size=(batch_size, self.latent_dim))
                noise = 1 / (1 + np.exp(-noise))
                real_y = np.ones((batch_size, 1))

                self.discriminator.trainable = False
                g_loss = self.dcgan.train_on_batch(noise, real_y)

                start += batch_size
        
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if checkpoint_path:
                self.save_model(checkpoint_path)

            if verbose and (epoch == 0 or(epoch + 1) % verbose == 0):
                self.generate(10, epoch+1, display=True)
        
        elapsed_time = time.time() - start_time
        hr = int(elapsed_time // 3600)
        elapsed_time %= 3600
        min = int(elapsed_time // 60)
        sec = int(elapsed_time % 60)
        print(f'Total training time: {hr}:{min:02d}:{sec:02d}')
        return d_losses, g_losses

    def generate(self, n_examples, epoch=None, display=False):
        noise = np.random.normal(size=(n_examples, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        if display:
            rows = n_examples // 5
            rows += 1 if n_examples % 5 > 0 else 0
            fig, ax = plt.subplots(rows, 5, figsize=(5, 3))
            fig.patch.set_facecolor('white')
            for indx in range(n_examples):
                img = gen_imgs[indx]
                img = self.preprocessor.inverse(img)
                img = img.astype(int)
                i, j = indx // 5, indx % 5
                ax[i, j].imshow(img, cmap=plt.cm.binary)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

            label = 'Epoch {0}'.format(epoch)
            fig.text(0.5, 0.04, label, ha='center')
            plt.show()
            
        return gen_imgs

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            
        disc_file = os.path.join(model_path, 'discrminator.h5')
        self.discriminator.save(disc_file)

        gen_file = os.path.join(model_path, 'generator.h5')
        self.generator.save(gen_file)

        dcgan_file = os.path.join(model_path, 'dcgan.h5')
        self.dcgan.save(dcgan_file)
