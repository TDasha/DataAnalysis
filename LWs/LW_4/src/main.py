# coding=utf-8

"""
Lab work # 4

Предсказание зарплаты по резюме.
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from time import time

trainData = pd.read_csv(r"../data/train.csv")

text = trainData['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

"""
TF-IDF

TF-IDF (от англ. TF — term frequency, IDF — inverse document frequency)
 — статистическая мера, используемая для оценки важности слова 
в контексте документа, являющегося частью коллекции документов 
или корпуса. Вес некоторого слова пропорционален количеству 
употребления этого слова в документе, и обратно пропорционален 
частоте употребления слова в других документах коллекции.
"""

vec = TfidfVectorizer(input='content', encoding='utf-8', analyzer='word', min_df=5)
X_train_text = vec.fit_transform(text)

trainData['LocationNormalized'].fillna('nan', inplace=True)
trainData['ContractTime'].fillna('nan', inplace=True)

"""
Dummy-кодирование или one-hot кодирование

Весь смысл категориальности теряется при простом сопастовлении признаку вещественого числа. 
Более того, появляются ложные интерпретации. 
Есть простейший метод, лишённый этого недостатка, его часто называют наивным / глупым (dummy) кодированием
или one-hot-кодированием. 
Для кодируемого категориального признака создаются N новых признаков, где N — число категорий. 
Каждый i-й новый признак — бинарный характеристический признак i-й категории.
"""

enc = DictVectorizer()
X_train_cat = enc.fit_transform(trainData[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_text, X_train_cat])

y_train = trainData['SalaryNormalized']

"""
Регрессия - частный случай задачи обучения с учителем, при котором целевая переменная принадлежит бесконечному 
подмножеству вещественной оси.

Ридж-регрессия или гребневая регрессия (англ. ridge regression) - это один из методов понижения размерности. 
Часто его применяют для борьбы с переизбыточностью данных, когда независимые переменные коррелируют друг с другом 
(т.е. имеет место мультиколлинеарность). Следствием этого является плохая обусловленность матрицы X^T X и неустойчивость
оценок коэффициентов регрессии. Оценки, например, могут иметь неправильный знак или значения, которые намного 
превосходят те, которые приемлемы из физических или практических соображений.

Применение гребневой регрессии нередко оправдывают тем, что это практический приём, с помощью которого при желании можно
получить меньшее значение среднего квадрата ошибки.

Метод стоит использовать, если:

* сильная обусловленность;
* сильно различаются собственные значения или некоторые из них близки к нулю;
* в матрице X есть почти линейно зависимые столбцы.
"""

model = Ridge(alpha=1.0)
start = time()
model.fit(X_train, y_train)
print("На обучение модели потрачено:", "%.6f" % (time() - start), "секунд", sep="\t")

testData = pd.read_csv(r"../data/test.csv")

text = testData['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

X_test_text = vec.transform(text)
X_test_cat = enc.transform(testData[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)

counter = 1
print("Ответы:")
for i in y_test:
    print(str(counter) + ":", "%.2f" % i, sep="\t")
    counter += 1
