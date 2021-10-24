import keras as k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as k
import h5py, json

TRAIN_SLICE = 600
TEST_SLICE = 0

class Preprocess:
    def __init__(self, data, features):
        self.df = data
        self.features_list = features
        # Ключевой известный результат
        self.output = ['Survived']
        # Преобразовываем данные
        self.raw_input = self.df[self.features_list]
        self.raw_output = self.df[self.output]
        # Преобразуем возраст в тип от 0 да 1
        # Преобразуем пол в 0==m 1==F
        # Преобразуем класс в массив, где первый класс - это [1, 0, 0], второй - [0, 1, 0], третий - [0, 0, 1]
        self.max_age = 100
        self.encoders = {
                "Age": lambda age: [age / self.max_age],
                "Sex": lambda sex: {"male" : [0], "female" : [1]}.get(sex),
                "Pclass": lambda plass: {1 : [1, 0, 0], 2 : [0, 1, 0], 3 : [0, 0, 1]}.get(plass),
                "Siblings/Spouses Aboard": lambda ss_aboard: [ss_aboard],
                "Parents/Children Aboard": lambda pc_aboard: [pc_aboard],
                "Survived": lambda s_value: [s_value]}

    def get_real_fact(self):
        return self.output

    def get_dataframe(self):
        return self.df

    def df_todict(self, df):
        result = dict()
        for column in df.columns:
            values = df[column].values
            result[column] = values
        return result

    def make_supervised(self, df):
        raw_input = df[self.features_list]
        raw_output = df[self.output]
        return {"input": self.df_todict(raw_input),
                "output": self.df_todict(raw_output)}

    def get_supervised(self):
        return self.make_supervised(self.df)

    def encode(self, data):
        vectors = []
        for data_name, data_values in data.items():
            encoded = list(map(self.encoders[data_name], data_values))
            vectors.append(encoded)
        formatted = []
        for vector_raw in list(zip(*vectors)):
            vector = []
            for elem in vector_raw:
                for e in elem:
                    vector.append(e)
            formatted.append(vector)
        return formatted

    def get_input_data(self):
        supervised = self.get_supervised()
        encd = self.encode(supervised['input'])
        return np.array(encd)
    
    def get_output_data(self):
        supervised = self.get_supervised()
        encd = self.encode(supervised['output'])
        return np.array(encd)

class Neuron():
    def __init__(self, data, features):
        self.Preprocessing = Preprocess(data=data, features=features) #NOSONAR
        self.df = data
        self.features_list = features
        self.input_data = self.Preprocessing.get_input_data()
        self.output_data = self.Preprocessing.get_output_data()
        self.train_x = self.input_data[:TRAIN_SLICE]
        self.train_y = self.output_data[:TRAIN_SLICE]

        self.test_x = self.input_data[TEST_SLICE:]
        self.test_y = self.output_data[TEST_SLICE:]

        self.model = k.Sequential()

    def _train_model(self):
        #Тренировка
        #Вход. units==5 так как мы используем 5 входных параметров
        self.model.add(k.layers.Dense(units=5, activation="relu"))
        #Выход. units==1 так как мы используем 1 выодныой параметр
        self.model.add(k.layers.Dense(units=1, activation="sigmoid"))
        self.model.compile(loss="mse", optimizer="sgd", metrics=['accuracy'])
        # Если тренируем
        fit_results = self.model.fit(x=self.train_x, y=self.train_y, epochs=1000, validation_split=0.2)
        fit_results.model.save('saved_models/razum.h5')

        # Рисуем графики
        show_graph = input('Показать график? y/n: ')
        if show_graph == 'y':
            plt.title("Losses train/validation")
            plt.plot(fit_results.history["loss"], label="Train")
            plt.plot(fit_results.history["val_loss"], label="Validation")
            plt.legend()
            plt.show()

            plt.title("Accuracies train/validation")
            plt.plot(fit_results.history["accuracy"], label="Train")
            plt.plot(fit_results.history["val_accuracy"], label="Validation")
            plt.legend()
            plt.show()

    def train(self):
        # Тренировка
        train = self._train_model() #NOSONAR

    def predict(self):
        self.model.built = True
        # Запускаем предикт
        predicted_test = k.models.load_model('saves_models/razum.h5').predict(self.test_x)
        # Формируем таблицу с предиктом
        real_data = self.Preprocessing.get_dataframe().iloc[TEST_SLICE:][self.features_list + self.Preprocessing.get_real_fact()]
        # Добавляем столбец Predict
        real_data['Predict'] = predicted_test
        return real_data

# Загружем данные
df = pd.read_csv('data/titanic.csv')
features_list = ['Age', 'Sex', 'Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
# Тренируем нейронку
to_train = input('Тренировать модель?: y/n ')
if to_train == 'y':
    train = Neuron(data=df, features=features_list).train()
# Вызываем нейронку
result = Neuron(data=df, features=features_list).predict()
print(result)
