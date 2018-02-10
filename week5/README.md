(c) HakOh 2018

## W5 - 경사하강법


#### 수치 미분 vs 해석적 미분
- 수치 미분 : 아주 작은 차분(n)으로 미분하는 것
- 해석적 미분 : 수식을 전개해서 미분하는 것

#### 수치미분 코드 구현하기

```python
def function_1(x):
   return 0.01 * x ** 2 + 0.1 * x

def numerical_diff(f,x):
   h = 1e-4 # 0.0001
   return (f(x+h) - f(x-h)) / (2*h)

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
```

#### 편미분
- 변수가 여럿인 함수에 대한 미분을 편미분이라 한다.

#### 편미분 코드 구현하기

```python
def function_2(x):
  return x[0]**2 + x[1]**2
  # 또는 return np.sum(x**2)

def function_tmp1(x0):
  return x0*x0 + 4.0 ** 2.0

numerical_diff(function_tmp1, 3.0)

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

numerical_diff(function_tmp2, 4.0)
```

#### 기울기
- 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 합니다.

#### 기울기 코드 구현하기

```python
def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

  for idx in range(x.size):
    tmp_val = x[idx]
    # f(x+h) 계산

    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h) 계산
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val #값 복원

  return grad
```
numerical_gradient(function_2, np.array([3.0, 4.0]))

numerical_gradient(function_2, np.array([0.0, 2.0]))

numerical_gradient(function_2, np.array([3.0, 0.0]))
```
