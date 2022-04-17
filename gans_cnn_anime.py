from ast import Num
from genericpath import exists
import sys
assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"



# Common imports
import numpy as np
import os
import numpy as np
import os
from PIL import Image

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)




def plot_multiple_images(images, epoch, n_cols=None):
    '''
    Plot multiple images in a grid
    '''
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    plt.savefig('.\\results\\anime_epoch' + str(epoch) + '.png')
    plt.close() 


def save_generated_images(epoch, generated_images, batch_size, count):
    '''
    Save all the generated images in a particluar batch of a particular epoch in a folder
    '''
    if not(exists('.\\generated_images\\')):
       os.mkdir('.\\generated_images\\')
    if not(exists('.\\generated_images\\' + str(epoch))):
       os.mkdir('.\\generated_images\\' + str(epoch))
    for i in range(generated_images.shape[0]):
        plt.imshow((generated_images[i]), cmap="binary")
        
        plt.axis("off")
        number = batch_size*count + i

        path = '.\\generated_images\\' + str(epoch) + '\\' + str(number) + '.png'

        plt.savefig(path)
        plt.close()
        #



def train_gan(gan, dataset, batch_size, codings_size, n_epochs=10):
    generator, discriminator = gan.layers
    count = 0
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))              # not shown in the book
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

            # save generated images
            #save_generated_images(epoch, generated_images, batch_size, count)
            count += 1
        plot_multiple_images(generated_images, epoch, 8)  
    generator.save("generator")
    plot_multiple_images(generated_images, epoch, 8)  




def load_images_from_path(path):
    images = []
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))
        img = img.resize((28,28))  # resize to (28,28)
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    
    return np.array(images)


if __name__ == "__main__":
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "images"
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "archive_anime", CHAPTER_ID)  # path to anime images
    #os.makedirs(IMAGES_PATH, exist_ok=True)

    if not tf.config.list_physical_devices('GPU'):
        print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
        if tf.test.is_built_with_cuda():
            print("But it looks like you have a CUDA device, so I'll use that.")
        else:
            print("But it looks like you don't have a CUDA device, so I'm not sure what to do next.")

    train_data = load_images_from_path(IMAGES_PATH)

    tf.random.set_seed(42)
    np.random.seed(42)

    codings_size = 100

    generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                    activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="SAME",
                                    activation="tanh"),
    ])
    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                            activation=keras.layers.LeakyReLU(0.2),
                            input_shape=[28, 28, 3]),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                            activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    gan = keras.models.Sequential([generator, discriminator])

    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

    train_dcgan = train_data.reshape(-1, 28, 28, 3) * 2. - 1. # reshape and rescale

    batch_size = 32

    dataset = tf.data.Dataset.from_tensor_slices(train_dcgan)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)



    train_gan(gan, dataset, batch_size, codings_size)