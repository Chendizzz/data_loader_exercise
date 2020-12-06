import pandas as pd
import numpy as np
import os
import matplotlib as plt
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
# print(dic)

file = Series.to_numpy(df['id'])
# print(file.shape)

file_name = [i+'.jpg' for i in file]
# print(file_name)

file_with_path = [os.path.join('D:\\ML_data\\dog_breed\\origin_train_data', i) for i in file_name]

'''seperate the dataset to a training set and a evaluation set
'''
file_train_path = file_with_path[:8000]
file_test_path = file_with_path[8000:]
# print(file_test_path)

'''save these paths with np.save(), as .npy file
'''
np.save('file_train_path.npy', file_train_path)
np.save('file_test_path.npy', file_test_path)

'''train labels
'''
number = []
for i in range(10222):
    number.append(  dic[ breed[i] ]  )
number = np.array(number)
number_train = number[:8000]
number_test = number[8000:]
np.save( "number_train.npy", number_train )
np.save( "number_test.npy" , number_test )

''' form a custom dataset
'''
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
'''use pil to read images
'''
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocessing = transforms.Compose([transforms.ToTensor(), normalize])


def default_loader(path):
    img_f = Image.open(path)
    print('img_f type: ', type(img_f))
    img_f = img_f.resize((224, 224))
    img_tensor = preprocessing(img_f)
    print(img_tensor.shape)
    return img_tensor


class trainset(Dataset):
    def __init__(self, loader=default_loader):

        self.images = file_train_path
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)

train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4, shuffle=True)

for step, (batch_x, batch_y) in enumerate(trainloader):
    print('step:', step, '| batch x size :', batch_x.size(), '| batch_y size:', batch_y.size())