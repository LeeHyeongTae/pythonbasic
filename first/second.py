#!/usr/bin/env python
# coding: utf-8

# In[4]:


# def 가 붙어 있으면 함수다. 클래스 외부에 있는 메서드(클래스와 관계 없음)라고 생각하면 된다.
def times(a, b):
    return a * b

print(times) 
#객체가 위치하는 메모리 주소값. C, C++스타일로 포인터. 자바스타일로 객체 나오는 숫자값은 근시대 컴퓨터 16자리 수
#이유는 포인터의 크기가 64비트 컴퓨터에서 8 byte이기 때문
#8바이트라는 것은 16진수로 표기했을 때 16자리
# 16진수 1자리는 4비트(bit)
# WebAssembly(웹 어셈블리) - 웹기술 + C++
print(times(3, 7))

#built-in 별도의 import없이도 사용할 수 있는 내장 함수들
#globals()라는 것은 파이썬이 구동되면서 활용할 수 있는 함수들의 리스트를 보여준다.
print(globals())


# In[5]:


# times()라는 함수의 객체 정보를 받아서 저장한다.
# 변수의 정의 = 특정한 데이터 타입이 저장되는 메모리 상의 공간 (state - 어느정도 무방)
pointerOfFunction = times #자바에서 인터페이스.
res = pointerOfFunction(7, 7)
print(res)


# In[6]:


def add(a, b):
    return a + b

# 현재 여기서는 pof 변수가 add 함수의 주소를 저장한다.
pof = add
# 그러므로 이것은 add(3, 7)과 같다.
res = pof(3, 7)
print("res =", res)

pof = times
res = pof(3, 7)
print("res =", res)
# 경우에 따라서 덧셈, 뺄셈, 곱셈 등 자유롭게 수행할 수 있게 해주는 기법
# 인터페이스를 사용하는 이유, RPG게임을 구현한다고 가정하면 여러 직업의 데미지 계산을 각각 만들면 관리포인트가 늘어나는데, 인터페이스를 사용하면
# 모든 데미지 계산은 해당 인터페이스를 가지고 처리하게 되어 일관성을 가지게 된다. (다형성)


# In[7]:


def intersect(prelist, postlist):
    retList = []
    for x in prelist:
        if x in postlist:
            retList.append(x)
    return retList

list1 = "Apple"
list2 = "onion"

print(intersect(list1, list2))
print(intersect(list1, ['H','A','M']))


# In[9]:


def swap(x, y):
    return y, x #내부적으로 한번 감싼 상태에서 리턴하게 됨.

print(swap(3, 7))

a, b = swap(33, 77)
print(a, b)

x = swap(333, 777)
print(x)
print(type(x))


# In[10]:


def change(x):
    x[0] = 'H'
    
wordlist = ['J', 'A', 'M']
print(wordlist)

change(wordlist)
print(wordlist)


# In[13]:


def change2(x):
    x = x[:] # x[:] 새로운 동일 객체를 만든다. 인자로 들어온 x와 새로만든 x가 분리된다. x를 변경하지 않고 값을 가공하고자 할때 사용.
    x[0] = 'H'
    print(x)
    
wordlist = ['J', 'A', 'M']
print(wordlist)
print("#######")
change2(wordlist)
print("#######")
print(wordlist)


# In[17]:


# 변수 glob을 선언함.
glob = 1
print(glob)

def xchGScope(x):
    # glob는 전역변수로 지정
    global glob 
    glob = 7
    return glob + x

print(xchGScope(3))
print(glob)


# In[19]:


def times(a = 10, b = 20):
    return a * b

print(times())
print(times(5))
print(times(3, 7))


# In[20]:


def connectURL(server, port):
    str = "http://" + server + ":" + port
    return str

name = "test.com"
service = "8080"

# 파라미터의 순서를 지키지 않아도 받을 수 있음.
print(connectURL("test.com","8000"))
print(connectURL(port = "8000", server = "test.com"))
print(connectURL("test.com", port = "8000"))
print(connectURL(name, service))


# In[21]:


# 가변 인자로 값을 처리하게 되면 튜플 타입이 된다.
def test(*args):
    print(type(args))
    
test(1, 2, 3)


# In[22]:


#*(에스테릭) - 포인터 x
# 파이선에서 가변인자를 받을 경우 *을 붙인다.
def union2(*ar):
    res = []
    #ar 튜플에서 가각의 요소를 item으로 뺀다.
    for item in ar:
        #item에 있는 글자를 한개씩 x로 뺀다.
        for x in item:
            #res라는 리스트와 값이 같은지 확인하고 같은 것이 없다면 X를 res리스트에 추가한다.
            if not x in res:
                res.append(x)
    return res

#중복 문자를 제외하고 사용된 모든 문자
print(union2("HAM", "EGG", "POTATO"))
print(union2("test", "tdd", "junit"))


# In[3]:


# 변수는 언제 할당 될까?
# Java int num; (선언만으로는 변수가 할당 되지 않는다.)
# 값을 집어넣어야 변수가 할당되면서 메모리가 잡힌다.
g = lambda x, y: x * y
print(g(2, 3))

print((lambda x: x * x)(3))

# 메모리를 할당 받지 않으면 완벽한 익명객체가 된다.
onemore = lambda x, y, z: x * y * z

print(onemore(2, 3, 4))

print(globals())


# In[4]:


#def로 선언한 함수들은 메모리구조상 Text라는 영역에 잡히게 된다. 가상 메모리 개념에 보면 Text|Data|Heap|Stack 에 배치됨.
#상수는 전부 Data
#new한 것들은 전부 Heap
#그 외의 변수들은 Stack에 배치 람다는 Stack에 있음.
#시스템언어들은 이것을 컴파일 시점에 링킹이란 작업을 통해서 겹치지 않게 조정한다.
#반면 인터프리팅 방식의 언어들은 이것을 Run-Time(실행중) 동적으로 조정한다.
# 조정하기 위해서 각 영역의 분리가 필요하다.
# 자바는 하이브리드 방식을 채택, javac는 컴파일러 역할, java는 인터프리터 역할


# In[5]:


help(print)


# In[6]:


import math
help(math)


# In[9]:


def plus(a, b):
    return a + b
#언더바 2개
plus.__doc__ = "a plus b"
help(plus)


# In[2]:


def factorial(n):
    """ n! 을 구하는 함수다.
    감마 함수가 적용되지 않아 
    1.1!, 1.3! 등을 구할 수 없다.
    또한 값은 0보다 커야한다.
    """
    if x < 0:
        print("error")
    elif x <= 1:
        return 1
    return x * factorial(x - 1)
help(factorial)


# In[6]:


# 문제 1.
# 1 ~ 100까지 숫자 중 2의 배수만 출력

for i in range(2, 101, 2):
    print(i)


# In[7]:


# numpy - Numerical Python - 수치해석을 위한 라이브러리
import numpy as np
# matplotlib - 그래프를 그리는 라이브러리
import matplotlib.pyplot as plt

# np.linspace - np 라이브러리 내부에 있는 linspace method
# linspace - Linear Space의 약자
# 증가폭이 일정한 값으로 0 ~ 5 사이의 숫자를 1001개 생성해
# 파라미터 첫번째 시작값(포함), 두번째 끝값(포함), 세번째 개수
# 그렇기 때문에 각 숫자간의 간격이 0.005
# 1001개 라는 개수를 샘플 개수
t = np.linspace(0, 5, 1001)
# y = 2 cos(10 * t)
# 진폭(amplitude) = 2
# 주기(period) = 주파수와 역수 관계
# 주파수(frequency) = 주기와 역수 관계
# 2 * pi * f * t = w * t
# w = 10
# 2 * pi * f = 10
# f = 10 / 2pi
# T(주기) = 2pi / 10
x1 = 2 * np.cos(10 * t)
#plt.subplots()는 그래프 폼을 만듦
#figsize = (6, 2.5)의 크기로 만듦
fig, ax = plt.subplots(figsize = (6, 2.5))
# 시간에 따른 cos 함수의 위치를 찍어보자.
ax.plot(t, x1)
#그래프 상에서 x의 범위는 0 ~ 5까지
ax.set_xlim(0, 5)
# x축
ax.set_xlabel('t(sec)')
ax.set_title('$x_1(t) = 2₩cos(10t)$')
# 화면상에 그래프가 보여진다.
fig.show()


# In[9]:


# y = x 를 그려보자.
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 1001)

y = x

fig, ax = plt.subplots(figsize = (3, 3))

ax.plot(x, y)

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.set_title('$y = x$')

fig.show()


# In[10]:


# 랜덤
import random
help(random)


# In[12]:


import random

cnt = 0
while cnt < 10:
    # randrange(시작값(포함), 끝값(포함x))
    # 위의 값 사이에서 랜덤값을 뽑아온다.
    # 1 ~ 6 사이의 랜덤값
    x = random.randrange(1, 5)
    print(x)
    cnt += 1


# In[ ]:


# 직삼각형의 대각선 길이를 구해보자.
import math

width = 6
height = 5

# 피타고라스 정리 : 대각선 길이 = 루트 ( 밑변^2 + 높이^2)

diag = math.sqrt(width * width + height * height)
print(diag)

diag = math.sqrt(width ** 2 + height ** 2)
print(diag)

diag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
print(diag)

