import pandas as pd
import numpy as np
import os

'''read the labels in the .csv file
'''
df = pd.read_csv('D:\\ML_data\\dog_breed\\labels.csv')
print(df.info())
print(df.head())

'''convert the labels to numpy.ndarray
'''
from pandas import Series, DataFrame
breed = df['breed']
breed_np = Series.to_numpy(breed)

print(type(breed_np))
print(breed_np.shape)
print(breed_np[0:5, ])

'''here set works similarly to np.unique
'''
# breed_set = set(breed_np)
breed_set = np.unique(breed_np)
print(len(breed_set))

breed_120_list = list(breed_set)
print(breed_120_list)

'''convert to a dictionary
'''
dic = {}
for i in range(len(breed_set)):
    dic[breed_120_list[i]] = i
print(dic)

file = Series.to_numpy(df['id'])
print(file.shape)

file_name = [i+'.jpg' for i in file]
print(file_name)

file_with_path = [os.path.join('D:\\ML_data\\dog_breed\\origin_train_data', i) for i in file_name]

'''seperate the dataset to a training set and a evaluation set
'''
file_train_path = file_with_path[:8000]
file_test_path = file_with_path[8000:]
print(file_test_path)

'''save these paths with np.save(), as .npy file
'''
np.save('file_train_path.npy', file_train_path)
np.save('file_test_path.npy', file_test_path)
