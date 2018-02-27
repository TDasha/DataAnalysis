"""
LabWork #2.

Распознование рукописных цифр из MNIST https://www.kaggle.com/c/digit-recognizer.

Main procedure.
"""
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import time

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""
Преобразуем метки в категории, пример 
3 === [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
"""
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# Создаем последовательную модель
model = Sequential()
"""
Добавляем 2 уровня нейронной сети сети.
Dense - тип сети, при котором все нейроны одного уровня соединены со всеми нейронами следующего уровня.
"""
model.add(Dense(400, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(200, activation="relu", kernel_initializer="normal"))
"""
800 нейронов, 784 входа (по кол-ву пикселей в входном изображении), распределение НОРМАЛЬНОЕ, relu = ф-ция активации.
Входные веса инициализируются случайным значениями с помощью НОРМАЛЬНОГО распределения
"""
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

# Необходимо скомпилировать модель модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
"""
loss - мера ошибки, optimizer - метод обучения (используем SGD === метод стохастического градиентного спуска.
Оптимизацию выполняем по метрике ТОЧНОСТЬ
"""
# Распечатали кратко характеристики модели
print(model.summary())

# Обучаем сеть
start_time = time.time()
model.fit(X_train, Y_train, batch_size=200, epochs=100, validation_split=0.1, verbose=2)
end_time = time.time()
print("Время обучения %s секунд" % (end_time - start_time))
"""
batch_size - минивыборка на основании которой определяется направление градиента в данном случае используем ее для
каждых 200-сот (веса меняем после каждой).
epochs - сколько раз обучаем сеть на одном и том же наборе данных.
verbose - флаг для вывода данных в консоль в процессе обучения
validation_split =  размер проверочной выборки validation set (для контроля в процессе обучения).
В данном случае 60000 * 0.2 = 12000.
"""
# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print(scores)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
