import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import detectors
import timm

os.makedirs('results',exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'models', 'my_resnet50_cifar10.pth')


cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar10_test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=cifar10_transform_test)
cifar10_testLoader = torch.utils.data.DataLoader(cifar10_test,batch_size=1,shuffle=True)

#모델 load
try:
    resnet50_cifar10 = timm.create_model("resnet50_cifar10", pretrained=True).to(device)
    resnet50_cifar10.eval()
    
    my_resnet50_cifar10 = models.resnet50(weights=None)
    my_resnet50_cifar10.fc = torch.nn.Linear(my_resnet50_cifar10.fc.in_features,10) #class 10개로 수정
    my_resnet50_cifar10.load_state_dict(torch.load(model_path))
    my_resnet50_cifar10.to(device)
    my_resnet50_cifar10.eval()
    
    Models = [resnet50_cifar10,my_resnet50_cifar10]
    print('load success')
    
except:
    print("load failed")
    
activated_neurons = {}

def deepxplore_loss(output,model_idx):
    loss_diff = 0
    cur_output = output[model_idx]
    for i,out in enumerate(output):
        if i != model_idx:
            loss_diff += torch.norm(cur_output - out,p=2)
    
    return -loss_diff

def gen_diff_pytorch(Models,input,iteration=50,lr=0.01):
    input = input.clone().detach().to(device).requires_grad_(True)
    
    for i in range(iteration):
        output = [model(input) for model in Models]
        preds = [torch.argmax(out,dim=1).item() for out in output]
        
        #두 모델의 예측이 달라짐
        if len(set(preds)) > 1:
            print(f"disagreement iteration :{i}, preds :{preds}")
            return input,True,preds
        
        loss = deepxplore_loss(output,0)
        loss.backward()
        
        with torch.no_grad():
            input -= lr * input.grad.sign()
            input.grad.zero_()
            input.clamp_(-1,1)
            
    preds = [torch.argmax(out,dim=1).item() for out in output]
    
    return input,False,preds

def unnormalize(t):
    return t*0.5 + 0.5

disagreed_cnt = 0
for i,(img,label) in enumerate(cifar10_testLoader):
    if disagreed_cnt >= 5:
        break
    
    img,label = cifar10_test[i]
    img = img.unsqueeze(0)
    adv_img,success,pred = gen_diff_pytorch(Models,img)
    
    if success:
        disagreed_cnt += 1
        
        
        ori = unnormalize(img.squeeze()).detach().cpu().permute(1,2,0).numpy()
        adv = unnormalize(adv_img.squeeze()).detach().cpu().permute(1,2,0).numpy()
        diff = np.abs(ori-adv)
        diff = diff / diff.max() if diff.max() > 0 else diff
        
        fig = plt.figure()
        fig.tight_layout()
                
        #원본 이미지
        ax1 = fig.add_subplot(131)
        ax1.set_title('Original')
        ax1.imshow(ori)
        
        #Adversial 이미지
        ax2 = fig.add_subplot(132)
        ax2.set_title('Adversarial')
        ax2.imshow(adv)
        
        #차이를 확대한 이미지
        ax3 = fig.add_subplot(133)
        ax3.set_title('Perturbation')
        ax3.imshow(diff)
        
        save = f'results/dissagreement_{i}.png'
        plt.savefig(save)
        plt.close()