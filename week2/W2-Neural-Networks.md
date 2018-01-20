(c) HakOh 2017

## W2 - 신경망 구성하기


#### 3층 신경망 구현하기
- init_network() 함수는 가중치와 편향을 초기화하고 딕셔너리 변수인 network에 저장합니다.
- forward() 함수는 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현합니다.


```python
#define sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#define identity_function
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print (y)
```
#### softmax 구현

```python
#softmax 구현

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

#softmax

def softmax(a):
    c = np.max(a)  
    exp_a = np.exp(a-c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

#### activation function
+ binary classification - 일반적으로 sigmoid 사용
+ multiclass classification - 일반적으로 softmax 사용

#### 소프트맥스 함수 구현시 주의점
+ 오버플로 - 표현할 수 있는 범위가 한정되어 너무 큰 값을 표현할 수 없는 문제.
입력 신호 중 최대값을 빼주면 문제 해결 가능


#### 아래 코드에서 X의 row=1, column=2인데 (2,)로 표시되는 이유는?

```python
X = np.array([1, 2])
print(X.shape)

#(2,)
```
