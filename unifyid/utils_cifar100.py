from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import gflags

FLAGS = gflags.FLAGS


def data_cifar100():
    """
    Preprocess MNIST dataset
    :return:
    """
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.reshape(X_train.shape[0], 3, FLAGS.img_rows, FLAGS.img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, FLAGS.img_rows, FLAGS.img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, FLAGS.nb_classes)
    Y_test = np_utils.to_categorical(y_test, FLAGS.nb_classes)
    return X_train, Y_train, X_test, Y_test


def model_cifar100():
    """
    Defines MNIST model using Keras sequential model
    :param tf_placeholder:
    :return:
    """
    model = Sequential()
    #conv block1
    model.add(Convolution2D(FLAGS.nb_filters, 3, 3,
                            border_mode='same',
                            input_shape=(3, FLAGS.img_rows, FLAGS.img_cols)))
    model.add(Activation('relu',name='rel1_1'))
    model.add(Convolution2D(FLAGS.nb_filters, 3, 3,border_mode='same'))
    model.add(Activation('relu',name='rel1_2'))
    model.add(MaxPooling2D(pool_size=(FLAGS.nb_pool, FLAGS.nb_pool),name='maxpool'))
    model.add(Dropout(0.25,name='dropout1'))
    #conv block2
    model.add(Convolution2D(FLAGS.nb_filters, 3, 3,
                            border_mode='same'))
    model.add(Activation('relu',name='rel2_1'))
    model.add(Convolution2D(FLAGS.nb_filters, 3, 3,border_mode='same'))
    model.add(Activation('relu',name='rel2_2'))
    model.add(MaxPooling2D(pool_size=(FLAGS.nb_pool, FLAGS.nb_pool),name='maxpool2'))
    model.add(Dropout(0.25,name='dropout2'))
    #conv block3
    model.add(Convolution2D(FLAGS.nb_filters*2, 3, 3,
                            border_mode='same'))
    model.add(Activation('relu',name='rel3_1'))
    model.add(Convolution2D(FLAGS.nb_filters*2, 3, 3,border_mode='same'))
    model.add(Activation('relu',name='rel3_2'))
    model.add(MaxPooling2D(pool_size=(FLAGS.nb_pool, FLAGS.nb_pool),name='maxpool3'))
    model.add(Dropout(0.25,name='dropout3'))
    #conv block4
    model.add(Convolution2D(FLAGS.nb_filters*2, 3, 3,
                            border_mode='same'))
    model.add(Activation('relu',name='rel4_1'))
    model.add(Convolution2D(FLAGS.nb_filters*2, 3, 3,border_mode='same'))
    model.add(Activation('relu',name='rel4_2'))
    model.add(MaxPooling2D(pool_size=(FLAGS.nb_pool, FLAGS.nb_pool),name='maxpool4'))
    model.add(Dropout(0.25,name='dropout4'))
    #FC block
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu',name='rel_last'))
    model.add(Dropout(0.5,name='droupout_last'))
    model.add(Dense(FLAGS.nb_classes))
    model.add(Activation('softmax',name='softmax'))

    return model

