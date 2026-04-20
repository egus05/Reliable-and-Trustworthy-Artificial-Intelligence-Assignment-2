<h1>신뢰할수 있는 인공지능 Assignment2 </h1>

CIFAR10 Dataset을 이용하여 두 모델에 대한 DeepXplore 시뮬레이션을 하는 프로그램

gen_diff.py의 핵심 로직을 test.py에 구현(tensorflow => pytorch)

Differential Loss(deepxplore_loss): 두 모델의 예측 값 차이를 L2 Norm으로 계산

Coverage Loss(loss_coverage): 특정 레이어의 뉴런 활성화 정도를 평균내어, 더 많은 뉴런이 자극 받도록 유도

Neuron Coverage(gen_diff_pytorch): register_forward_hook을 사용해 모델의 특정지멍의 활성화 값을 추정, 출력이 0보다 큰 뉴련을 활성화로 판단




resnet50_cifar10_A: 기본 가중치 초기화 + SGD Optimizer + 기본 데이터 증강(flip)


resnet50_cifar10_B: Kaiming He 초기화 + AdamW Optimizer + 강력한 데이터 증강 (Color Jitter)


<h2>구성 요소</h2>
test.py: 두 모델을 불러와서 DeepXplore의 모델 테스팅 기법을 적용해 두 모델이 다른 판단을 내리도록 공격하고,

뉴런 커버리지를 측정하는 프로그램


requirement.txt: 프로젝트 실행을 위해 필요한 라이브러리 목록


results: 두 모델이 다르게 판단 한 이미지가 저장되는 폴더


model: 사전학습된 모델의 가중치(pth)저장 폴더


report: 적대적 공격과 테스팅 기법에 대한 에세이


<h2>실행방법</h2>
pip install -r requirements.txt #필수 라이브러리 설치
python test.py #프로그램 실행

<h2>예상 결과</h2>
1. 다르다고 분류한 이미지의 클래스와 반복횟수

2. 뉴런 커버리지(%)

3. 다르다고 분류한 이미지(png)
