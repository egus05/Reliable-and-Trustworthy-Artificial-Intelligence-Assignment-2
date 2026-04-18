import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

os.makedirs('results',exist_ok=True)

model1 = load_model('./models/resnet50_cifar10.h5')
model2 = load_model('./models/resnet50model1.h5')
models = [model1,model2]


def deepxplore_loss(output,model_idx):
    loss_diff = 0
    cur_output = output[model_idx]
    for i,out in enumerate(output):
        if i != model_idx:
            loss_diff += tf.norm(cur_output - out,p=2)
    
    return -loss_diff

def gen_diff(models,input,iteration=50,lr=0.01):
    img = tf.convert_to_tensor(input,dtype=tf.float32)
    
    for i in range(iteration):
        with tf.GradientTape() as tape:
            tape.watch(img)
            preds = [model(img) for model in models] #예측값 계산
            
            loss = deepxplore_loss()
            
            grads = tape.gradient(loss,img)
            img = img - lr*tf.sign(grads)
            img = tf.clip_by_value(img,0,1) #이미지 범위 제한(0~1)
            
    return img.numpy()

disagreed_cnt = 0 #이미지 5개 카운트
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data() #cifar10 이미지 로드
x_test = x_test.astype('float32') / 255.0 #정규화

for i,(img,label) in range(len(x_test)):
    if disagreed_cnt >= 5:
        break
    
    ori = x_test[i:i+1] # 4D로 변환
    adv = gen_diff(models,img)
    
    pred1 = np.argmax(model1.predict(adv))
    pred2 = np.argmax(model2.predict(adv))
    
    #두 예측 값이 다른 경우
    if pred1 != pred2:
        disagreed_cnt += 1
        
        
     
        fig = plt.figure()
        fig.tight_layout()
                
        #원본 이미지
        ax1 = fig.add_subplot(131)
        ax1.set_title('Original')
        ax1.imshow(ori[0])
        
        #Adversial 이미지
        ax2 = fig.add_subplot(132)
        ax2.set_title('Adversarial')
        ax2.imshow(adv[0])
        
        save = f'results/dissagreement_{i}.png'
        plt.savefig(save)
        plt.close()