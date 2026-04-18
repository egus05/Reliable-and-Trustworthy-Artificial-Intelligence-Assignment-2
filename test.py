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
modelA_path = os.path.join(base_path, 'models', 'resnet50_cifar10_A.pth')
modelB_path = os.path.join(base_path, 'models', 'resnet50_cifar10_B.pth')

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar10_test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=cifar10_transform_test)
cifar10_testLoader = torch.utils.data.DataLoader(cifar10_test,batch_size=1,shuffle=True)

def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features,256),
        torch.nn.ReLU(),
        torch.nn.Linear(256,10)
    )
    model.load_state_dict(torch.load(path,map_location=device))
    model.to(device)
    model.eval()
    return model

#모델 load
try:
    """
    #resnet50_cifar10 = timm.create_model("resnet50_cifar10", pretrained=True).to(device)
    #resnet50_cifar10.eval()
    
    my_resnet50_cifar10 = models.resnet50(weights=None)
    my_resnet50_cifar10.fc = torch.nn.Linear(my_resnet50_cifar10.fc.in_features,10) #class 10개로 수정
    my_resnet50_cifar10.load_state_dict(torch.load(modelA_path))
    my_resnet50_cifar10.to(device)
    my_resnet50_cifar10.eval()
    """
    
    modelA = load_model(modelA_path)
    modelB = load_model(modelB_path)
    
    Models = [modelA,modelB]
    print('load success')
    
except Exception as e:
    print(f"load failed {e}")
    exit()
    
activated_neurons = set() #활성화 된 뉴런을 저장
current_layer_output = None

#active된 뉴런을 탐지
def coverage_hook(module,input,output):
    global current_layer_output
    current_layer_output = output
    
    active = (output > 0).nonzero(as_tuple=False) #coverage 측정용
    for idx in active:
        activated_neurons.add(tuple(idx[1:].tolist()))

def deepxplore_loss(output,model_idx):
    loss_diff = 0
    lambda_val = 0.1
    cur_output = output[model_idx]
    for i,out in enumerate(output):
        if i != model_idx:
            loss_diff += torch.norm(cur_output - out,p=2)
            
    loss_coverage = torch.mean(current_layer_output)
    
    return -(loss_diff + lambda_val * loss_coverage)

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

Models[0].layer4[-1].register_forward_hook(coverage_hook)
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
        ax1.set_xlabel(f'predict : {pred[0]}')
        ax1.imshow(ori)
        
        #Adversial 이미지
        ax2 = fig.add_subplot(132)
        ax2.set_title('Adversarial')
        ax2.set_xlabel(f'predict : {pred[1]}')
        ax2.imshow(adv)
        
        #차이를 확대한 이미지
        ax3 = fig.add_subplot(133)
        ax3.set_title('Perturbation')
        ax3.imshow(diff)
        
        save = f'results/dissagreement_{i}.png'
        plt.savefig(save)
        plt.close()
    
    
total_nuerons = 2048*4*4
coverage = len(activated_neurons) / total_nuerons
print(f"Current Neuron coverage {coverage:.2f}")