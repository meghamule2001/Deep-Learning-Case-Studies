
#####################
# imports
#####################

import numpy as np

# import dataset
from keras.datasets import mnist

# importing the keras layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model

line = "*"*30

# load dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
print(line)

# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))
print(line)

# compute the number of labels
# correctly shape and format data
num_labels = len(np.unique(y_train))
print("computed number of labels :",num_labels)
print(line)

# convert to one-hot vector
y_train = to_categorical(y_train)
print("y_train after encoding")
print(y_train)
print(line)
y_test = to_categorical(y_test)
print("y_test after encoding")
print(y_test)
print(line)

# image dimensions assumed size
image_size = x_train.shape[1]
input_size = image_size * image_size
print("Image dimensions Assumed size in sq format : ", input_size)

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45

model = Sequential()
# model is a 3-layer MLP with ReLU and dropout after each layer
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
print(line)
print("Summary after applying Sequential model : ")
print(model.summary())
#plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
ann = model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))

import matplotlib as plt
plt.plot(ann.history['acc'])
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy plot")
plt.show()
