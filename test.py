import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models

cifar10_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar10_train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=cifar10_transform_train)
cifar10_test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=cifar10_transform_test)

resnet1 = models.resnet18(weights='DEFAULT')
resnet1.fc = torch.nn.Linear(resnet1.fc.in_features,10) #클래스 10개로 변경

resnet2 = models.resnet18(weights='DEFAULT')
resnet2.fc = torch.nn.Linear(resnet2.fc.in_features,10) #클래스 10개로 변경