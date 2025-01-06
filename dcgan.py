import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
    def __init__(self, input_shape, lr=2e-4, latent_dim=100, model_path=None):
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        if model_path is None:
            self.generator = self.__generator()
            self.discriminator = self.__discriminator(channels=input_shape[2])
            self.dcgan = self.__dcgan()
        else:
            disc_file = os.path.join(model_path, 'discriminator.h5')
            self.discriminator = keras.load_model(disc_file)

            gen_file = os.path.join(model_path, 'generator.h5')
            self.generator = keras.load_model(gen_file)

            dcgan_file = os.path.join(model_path, 'dcgan.h5')
            self.dcgan = keras.load_model(dcgan_file)
        self.__compile_models()

    def __generator(self):
        generator = keras.Sequential(
            [
                keras.Input(shape=(1, 1, self.latent_dim)),
                layers.Reshape((self.latent_dim,)),
                layers.Dense(8 * 8 * 128),
                layers.Reshape((8, 8, 128)),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
            ],
            name="generator",
        )
        return generator

    def __discriminator(self, channels=1):
        discriminator = keras.Sequential(
            [
                keras.Input(shape=(64, 64, channels)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        return discriminator

    def __dcgan(self):
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(1, 1, self.latent_dim))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        dcgan = keras.Model(gan_input, gan_output)
        return dcgan
    
    def __compile_models(self):
        """Compiles the discriminator and DCGAN models."""
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))  # Compile the generator
        self.dcgan.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    
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
