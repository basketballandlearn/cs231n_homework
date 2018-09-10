import torch
import numpy 
import operator
import torch.nn as nn
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
"""
a = torch.Tensor([[5,2,3],[5,5,5]])
b = torch.Tensor([[4,7,6]])
print(a.view(1, -1))
print(torch.sum(a, dim=1))
print(b.size())
print(torch.sort(b, dim=1))
"""
"""
b = [0]
print(b[0])
a = numpy.array([2,1,3,6,5])
print(a.argsort())

"""
input_size = 28*28

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root = './data/', 
                           train = True, 
                           #transform = transforms.ToTensor(), 
                           download = True) 
test_dataset = datasets.MNIST(root = './data/', 
                           train = False, 
                           #transform = transforms.ToTensor(), 
                           download = False)

def knn_classify(inputs, k=5):
    
    distance = []
    labels = []
    labelcount = {}
    
    for i in range(60000):
        train_set = train_dataset[i][0]
        train_set = transform(train_set)
        train_set = train_set.view(1, -1).numpy()

        dis1 = inputs - train_set
        
        dis2 = dis1**2
        dis3 = dis2.sum(axis=1)
        dis4 = dis3**0.5
        
        distance.append(dis4)
        labels.append(train_dataset[i][1])
    
    #labels = labels
    
    distance = numpy.array(distance)
    dis = []
    for r in range(60000):
        dis.append(distance[r][0])
    
    sortdis = numpy.array(dis).argsort()
    
    for m in range(k):
        index = sortdis[m]
        vote = labels[index]
        labelcount[vote] = labelcount.get(vote,0) + 1
    outputs = sorted(labelcount.items(),key=operator.itemgetter(1),reverse=True)
    return outputs[0][0]

total = 0.0
right = 0.0
    
for n in range(1000):

    test_set = test_dataset[n][0]
    test_set = transform(test_set)
    test_set = test_set.view(1, -1).numpy()
    
    label = test_dataset[n][1]

    print("预测为：",knn_classify(test_set).data)
    print("实际为：",label.data)
    if(label == knn_classify(test_set)):
        right += 1
    total =+ 1
acc = right/total
print('acc:',acc) 
    
    








