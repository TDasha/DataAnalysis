import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd
import os

"""
  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign = доброкачественная, 4 for malignant = злокачественная)
"""
print_end = "\n----------------------------------------------------------------------------------------------\n"
dir_path = os.path.dirname(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)))
df = pd.read_csv(dir_path + '\\data\\train.csv')
test_df = pd.read_csv(dir_path + '\\data\\test.csv')
# print(df, test_df, end=print_end, sep=print_end)
df.replace('?', -99999, inplace=True)
test_df.replace('?', -99999, inplace=True)
# некоторые ячейки в датасете пустые, присвоим им сильно отличающееся значение. сделали ли их статистическими выбросами
# print(df, end=print_end)
df.drop(['#1'], 1, inplace=True)
test_df.drop(['#1'], 1, inplace=True)
# вычеркнули столбец id, он не нужен
# print(df, test_df, end=print_end, sep=print_end)
X = np.array(df.drop(['#11'], 1))
y = np.array(df['#11'])
testX = np.array(test_df.drop(['#11'], 1))
testY = np.array(test_df['#11'])
# отделили медицинские данные от прогнозов в обоих датафреймах
# print(X, y, end=print_end, sep=print_end)
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train, X_test, y_train, y_test, sep=print_end, end=print_end)
clf = neighbors.KNeighborsClassifier()
# классификатор kNN = k Ближайших Соседей
clf.fit(X_train, y_train)
print(clf.predict(testX))
# обучили и смотрим точность
accuracy = clf.score(testX, testY)
print("Accuracy:", "%.6f" % accuracy, sep=" ")
# еще данные и прогнозы по ним
# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measures)
# print(prediction)
