from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import os
import h5py
import numpy as np


def addCNNlayer(model, n_filter, n_conv, n_pool, input_height=None, input_width=None, color=True):
    """adds a Convolution filter and a maxpooling layer. Optional params for input layer
    """
    n_color = 3 if color else 1
    if input_height and input_width:
        model.add(
            Convolution2D(
                n_filter,
                n_conv,
                n_conv,
                input_shape=(n_color, input_height, input_width)
            )
        )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
    else:
        model.add(
            Convolution2D(
                n_filter,
                n_conv,
                n_conv,
            )
        )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))


# path to the model weights file.
weights_path = 'lfs_tracked/vgg16_weights.h5'
top_model_weights_path = 'lfs_tracked/bottleneck_fc_model.h5'
# path to data files
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

# input image size
height = 150
width = 150


def save_bottleneck_features(img_width, img_height, weights_path, train_data_dir, validation_data_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load VGG16 network weights
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # ignore fully connected layers
            break
        layer = f['layer_{}'.format(k)]
        weights = [layer['param_{}'.format(param)] for param in range(layer.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('VGG16 model loaded')

    # save network output for training and validation sets to files for later use
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('lfs_tracked/bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('lfs_tracked/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    print('bottleneck features saved to file')


# save_bottleneck_features(width, height, weights_path, train_data_dir, validation_data_dir)

train_data = np.load(open('lfs_tracked/bottleneck_features_train.npy'))
train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
validation_data = np.load(open('lfs_tracked/bottleneck_features_validation.npy'))
validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

# full connected layer added to VGG16 convolution layers
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    train_labels,
    nb_epoch=nb_epoch,
    batch_size=32,
    validation_data=(validation_data, validation_labels),
)

model.save_weights(top_model_weights_path)
"""
# fully connected layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)

# data preparation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),     # all images resized to 150 x 150
    batch_size=32,
    class_mode='binary',
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),     # all images resized to 150 x 150
    batch_size=32,
    class_mode='binary',
)

model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=1,
    validation_data=validation_generator,
    nb_val_samples=800
)

model.save_weights('first_try.h5')
"""
