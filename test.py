import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import detectors
import timm

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'models', 'my_resnet50_cifar10.pth')

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

#모델 load
try:
    resnet50_cifar10 = timm.create_model("resnet50_cifar10", pretrained=True)
    resnet50_cifar10.eval()
    
    my_resnet50_cifar10 = models.resnet50(weights=None)
    my_resnet50_cifar10.fc = torch.nn.Linear(my_resnet50_cifar10.fc.in_features,10) #class 10개로 수정
    my_resnet50_cifar10.load_state_dict(torch.load(model_path))
    my_resnet50_cifar10.eval()
    
    print('load success')
    
except:
    print("load failed")