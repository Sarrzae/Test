import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np

###선형 회귀분석 모델 작성###

#데이터 준비
num_data=1000 #사용 할 데이터의 수
num_epoch=10000 #경사하강법 반복 횟수

x=init.uniform_(torch.Tensor(num_data,1),-15,15) #x값을 -15에서 15사이의 무작위 숫자들로 초기화
noise=init.normal_(torch.FloatTensor(num_data,1),mean=0,std=1) #평균0, 표준편차1인 가우시안 분포 노이즈
y=x**2+3 #x에 대한 종속 변수로 값이 3에서 228사이에 분포
y_noise=y+noise #현실성 반영을 위한 y값에 노이즈 추가

'''
#그래프 생성
# figure의 크기를 지정해줍니다.
plt.figure(figsize=(10,10))

# x축에는 x를 사용하고 y축에는 y_noise를 사용해 scatter plot 해줍니다.
# 이때 점의 크기는 7, 점의 색상은 회색으로 임의로 지정했습니다.
plt.scatter(x.numpy(),y_noise.numpy(),s=7,c="gray")

# figure의 x,y 축 범위를 지정해줍니다.
plt.axis([-12, 12, -25, 25])

# figure를 출력합니다.
plt.show()
'''

#선형회귀 모델 생성
# 아래 코드는 특성의 개수가 1 -> 6 -> 10 -> 6 -> 1개로 변하는 인공신경망
# 또한 선형변환 이후 활성화 함수를 넣어 비선형성이 생기도록 함
model=nn.Sequential(
      nn.Linear(1,6),
      nn.ReLU(),
      nn.Linear(6,10),
      nn.ReLU(),
      nn.Linear(10,6),
      nn.ReLU(),
      nn.Linear(6,1),
     )#Sequential class는 nn.Linear, nn.LeLU와 같은 모듈을 인수로 받아서 순서대로 정렬.
      #입력값이 들어오면 이 순서대로 모듈을 실행하여 결과 값을 리턴


loss_func=nn.L1Loss() #모델에서 나온 결과와 y_noise(실제 결과)와의 차이를 구하기 위해 L1손실함수(차이의 절대값의 평균) 사용
optimizer=optim.SGD(model.parameters(),lr=0.0002) #경사하강법으로 SGD옵티마이저 사용, model.parameters()를 통해 선형회귀 모델의 변수w와 b를 전달
                                                #lr=학습률
#print(model.weight, model.bias) #w, b값 확인

#학습 및 중간 확인

# 손실이 어떻게 변하는지 확인하기 위해 loss_arr를 만들어 기록
loss_arr =[]

# 또한 목표값은 y_noise로 지정
label = y_noise

#num_epoch 수(500) 만큼 학습 반복
for i in range(num_epoch):
    optimizer.zero_grad() #각 학습 시작 시, 이전에 계산했던 기울기 값을 0으로 초기화
    output=model(x) #선형회귀모델에 x값 전달

    loss=loss_func(output,label) #L1손실함수의 정의에 따라 output과 y_noise(label)의 차이를 loss에 저장
    loss.backward() #손실에 대한 기울기 계산
    optimizer.step() #인수로 들어갔던 model.parameters()에서 리턴되는 변수들의 기울기에 학습률 0.01을 곱하여 빼줌으로써 업데이트

    if i%100 ==0:
        print(i,"epoch")
        print("loss= ",loss.data) #손실 값 출력
        for param in model.parameters():
            print(param)
        '''
        # 현재 연산 그래프에 속해있는 x, output 값을 detach를 통해 분리하고, 텐서를 넘파이 배열로 바꿔서 plt.scatter에 전달
        plt.scatter(x.detach().numpy(), output.detach().numpy())
        plt.axis([-10, 10, -30, 30])
        plt.show()
        print(loss.data)
        '''
    # 손실을 loss_arr에 추가
    loss_arr.append(loss.detach().numpy())

#손실 그래프 표시
plt.plot(loss_arr)
plt.show()

#학습된 모델의 결과값과 실제 목표값의 비교
plt.figure(figsize=(10,10))
plt.scatter(x.detach().numpy(),y_noise,label="Original Data")
plt.scatter(x.detach().numpy(),output.detach().numpy(),label="Model Output")
plt.legend()
plt.show()