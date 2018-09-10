# 使用K近邻算法进行二维数字二分类

import numpy as np
import operator

def make_dataset():
    train_set = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = [1,1,0,0]
    return train_set, labels

train_set, labels = make_dataset()

def knn_classify(inputs, k=3):
  labelcount = {}
  raw = train_set.shape[0]
  inputs = np.tile(inputs,(raw,1))
  dis1 = inputs - train_set
  dis2 = dis1**2
  dis3 = dis2.sum(axis=1)
  dis4 = dis3**0.5
  sortdis = dis4.argsort()
  for i in range(k):
    vote = labels[sortdis[i]]
    labelcount[vote] = labelcount.get(vote,0) + 1
  outputs = sorted(labelcount.items(),key=operator.itemgetter(1),reverse=True)
  
  return outputs[0][0]

a = knn_classify([5,9])
print(a)

"""
#print(train_set.shape) 
#a = [2,3]
#a_tile = np.tile(a,(4,1))
#print(a_tile)

a = np.array([2,3])
a = a**2

a = np.array([[2,3]])
a = a.sum(axis=1)
print(a)

labelcount = {'a':2,'b':4}

p = labelcount.get('a',0) + 1
print(p)

labelcount = {'a':2,'b':5,'c':1}
outputs = sorted(labelcount.items(),key=operator.itemgetter(1),reverse=True)
print(outputs)

"""
