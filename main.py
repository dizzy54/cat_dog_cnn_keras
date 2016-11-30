from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


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


# input image size
height = 150
width = 150
# number of convolution filters
n_filter = 32
# filter size (n x n)
n_conv = 3
# pooling size (n x n)
n_pool = 2
model = Sequential()
addCNNlayer(
    model,
    n_filter=n_filter,
    n_conv=n_conv,
    n_pool=n_pool,
    input_height=height,
    input_width=width,
)

addCNNlayer(
    model,
    n_filter=n_filter,
    n_conv=n_conv,
    n_pool=n_pool,
)

n_filter = 2 * n_filter
addCNNlayer(
    model,
    n_filter=n_filter,
    n_conv=n_conv,
    n_pool=n_pool,
)

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