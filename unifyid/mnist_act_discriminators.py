import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense

def LinearSVC(input_shape):
	model = Sequential()
	model.add(Dense(2, input_shape=input_shape))
	return model
