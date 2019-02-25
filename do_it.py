import pandas as pd
import numpy as np
from keras import layers
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


train = pd.read_csv("C://Users//RA40024262//Downloads//digit-recognizer//train.csv")
test = pd.read_csv("C://Users//RA40024262//Downloads//digit-recognizer//test.csv")

y_train_org = train.iloc[:, 0].copy()
##length, width, channel
x_train, y_train = train.iloc[:, 1:].values.reshape(len(train), 28, 28, 1), train.iloc[:, 0].values
cat_y_train =  np.array(keras.utils.to_categorical(y_train, num_classes = 10))
model = Sequential()
model.add(Convolution2D(32,3,padding='same',
                    data_format='channels_last', activation='relu',input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(32,3,padding='same',
                    data_format='channels_last', activation='relu',input_shape = (28, 28, 1)))
model.add(Convolution2D(32,3,padding='same',
                    data_format='channels_last', activation='relu',input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
model.fit(x_train, cat_y_train, batch_size = 32, epochs = 20,    verbose = 0)
score = model.evaluate(x_train, cat_y_train, verbose = 0)
print(f'categorical corssentropy loss {score[0]}, accuracy {score[1]}')
pred = model.predict(x_train)
pred_org = []
for each in pred:
    pred_org.append(np.argmax(each))
plt.scatter(y_train, pred_org, label = "pred")
plt.xlabel('y_train')
plt.ylabel('predicted')
plt.legend()
plt.show()
pred = model.predict(test.iloc[:, :].values.reshape(len(test), 28, 28, 1))
pred_org = []
for each in pred:
    pred_org.append(np.argmax(each))
pd.DataFrame(pred_org).to_csv('submit.csv', index = True)














