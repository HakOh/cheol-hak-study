(c) HakOh 2018

## W3 - 배치 처리

** batch **

여러 데이터를 하나로 묶은 입력 데이터를 배치(batch)라 합니다.


** 배치 처리를 하는 이유 **

1. 넘파이 라이브러리의 큰 배열을 효율적으로 처리할 수 있기 때문이다.
2. CPU/GPU 측면에서도 큰 배열을 효율적으로 처리할 수 있기 때문이다.

** 배치 처리 코드 **

```python
# 신경망에 의한 추론수행, 정확도 평가
# load data
# x : input, t = x의 label 데이터
x, t = get_data()
# init_network 함수 불러옴 (week1 참조)
network = init_network()

batch_size = 100 #배치 크기
accuracy_cnt = 0
#batch_size=100이므로 x[0:100],x[100:200],..와 같이 앞에서 부터 100장씩 묶어서 꺼냄
for i in range(0, len(x), batch_size):
    # x를 batch_size만큼 가져온다
    x_batch = x[i:i+batch_size]
    # x_batch를 prediction 해서 나온 100x10 score 벡터
    y_batch = predict(network, x_batch)
    # x_batch 각 원소 데이터에서 가장 확률이 높은 clas를 나타낸 벡터
    p = np.argmax(y_batch, axis = 1)
    # p랑 실제 정답인 t_batch의 일치하는지 확인하여 정확도 측정
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print ("Accuracy:" + str(float(accuracy_cnt / len(x))))
```

![batch](/images/batch.jpeg)

+ `x_batch.shape` : (100, 784)
+ `y_batch.shape` : (100, 10)
+ `p = np.argmax(y_batch, axis = 1)` : np.argmax((100, 10), column per row)
+ `p.shape` : (100,)

## Further study

+ **(not yet)** 피클로 저장하는 법 해보기
+ **(not yet)** 부모 디렉토리의 부모 디렉토리에 있는 파이썬 파일 import하는 법
+ **(solved)** 네트워크를 짤 때 뉴런의 개수가 어떤 의도를 가진 것인지 의미 파악하기 == input feature를 뉴런 개수만큼 압축하거나 더 세밀하게 나눈다.
+ **(solved)** X의 row=1, column=2 인데 (2,)으로 표시되는 이유? == 1차원 칼럼벡터나 1차원 로우벡터는 (k,)으로 표시한다.
+ **(solved)** 출력층의 활성화 함수에서 binary classification에서는 sigmoid를 쓰고, multiclass classification에서는 softmax를 사용하는 이유? == 생각해보니 softmax는 sigmoid 보다 exp 연산을 더 많이 하므로 연산이 더 비싸서 binary에서는 sigmoid를 쓰는 것 같다.
+ **(solved)** 학습 단계에서 softmax 함수를 굳이 적용하는 이유가 무엇일까? == 학습시 loss를 계산해서 optimize를 할 때 카테고리 데이터는 score로 loss를 최적화하기보다는 각 카테고리에 해당할 확률을 계산해서 loss를 최적화하는 것이 논리적으로 더 맞다.
+ **(solved)** 추론 단계에서 softmax 함수를 생략해도 되는 이유는? == 학습 단계와 달리 loss를 구해서 optimize할 필요가 없고 단지 score 벡터의 argmax만 구하면 되기 때문이다.
