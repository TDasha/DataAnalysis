"""
Lab Work #3

Прогноз качества вин.
http://archive.ics.uci.edu/ml/datasets/Wine%2BQuality

Нейронная сеть
"""
from time import time
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)

print_end = "\n----------------------------------------------------------------------------------------------\n"

trainData = pd.read_csv("../data/train.csv", sep=";")
testData = pd.read_csv("../data/test.csv", sep=";")

trainX = np.array(trainData.drop(['#12'], 1))
trainY = np.array(trainData['#12'])
testX = np.array(testData.drop(['#12'], 1))
testY = np.array(testData['#12'])

mean = trainX.mean(axis=0)
std = trainX.std(axis=0)
trainX -= mean
trainX /= std
mean = testX.mean(axis=0)
std = testX.std(axis=0)
testX -= mean
testX /= std

model = Sequential()
model.add(Dense(11, activation='relu', input_shape=(trainX.shape[1],),  kernel_initializer="normal"))
model.add(Dense(1))
model.compile(optimizer='ADAM', loss='mse', metrics=['mae'])

start_time = time()
model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=2)
print("Время обучения :", time() - start_time, "секунд", sep=" ")

mse, mae = model.evaluate(testX, testY, verbose=0)
print("Средняя абсолютная ошибка :", mae)

# predict = model.predict(testX)
# print("Предсказанно:", predict[1][0], ", правильно:", testY[1])
