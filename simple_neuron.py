# Импортируем библиотеки
import pandas as pd
import numpy as np
import joblib as jbl
import matplotlib as plt
import tensorflow as tnsflow
import sklearn as skl

# Функция сигмоида (необходима для определения значения весов)
def sigmoid(x, der=False):
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Входные данные
input_data = np.array([ [0, 0, 1],
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0] ])

# Выходные данные (результат берется из левого стобца)
output_data = np.array([[0, 1, 0, 0]]).T # T - функция переноса

# Делаем случайные числа более определенными
np.random.seed(1)

# Первый синапс
# Инициализируем веса случайным образом со средним 0
# Его размерность 3 на 1 тк 3 столбца мы подаем на вход (input_data) и даем 1 на выход (output_data)
syn0 = 2 * np.random.random((3, 1)) - 1

# Обучение
l1 = []

for iter in range(100000):
    l0 = input_data
    l1 = sigmoid(np.dot(l0, syn0))
    l1_error = output_data - l1
    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

# Выходные данные после тренировки
print(f'Выходные данные после тренировки:\n{l1}')

# Проверяем нейронную сеть
test_input = np.array([1, 0, 1])
new_layer = sigmoid(np.dot(test_input, syn0))
print(f'\nВход: {test_input}\nВыход: {new_layer}')