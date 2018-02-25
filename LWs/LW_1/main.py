#import numpy as np
#from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
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
  11. Class:                        (2 for benign, 4 for malignant)
"""
print_end = "\n----------------------------------------------------------------------------------------------\n"
df = pd.read_csv('D:\\IDA1\\2_task_breast_cancer\\train.csv')
#print(df, end=print_end)
df.replace('?', -99999, inplace=True) # некоторые ячейки в датасете пустые, присвоим им сильно отличающееся значение
#сделаем выбросами?
#print(df, end=print_end)
df.drop(['#1'], 1, inplace=True)  # вычеркнули столбец id, он не нужен
#print(df, end=print_end)
