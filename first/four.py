#!/usr/bin/env python
# coding: utf-8

# In[12]:


import time
import multiprocessing

def countDown(x):
    while True:
        if x == 0:
            break
        print("CountDown ... %d" % x)
        x -=1
        time.sleep(1) # 1초간격으로 CPU를 실행

p = multiprocessing.Process(target = countDown, args = (5, ))
p.start()
print(p)


# In[7]:


import time
import multiprocessing

#우리가 프로세스로 만드려고 하는 함수
def process(msg, interval):
    while True:
        print("im working ... : %s" % msg)
        time.sleep(interval)
#프로세스로 동작할 녀석이 process 함수임을 target으로 알려준다.
#그리고 msg와 interval에 인자(파라미터)를 전달
# msg = "pX", interval은 대기시간
# 전부 부모 프로세스의 자식들
p1 = multiprocessing.Process(target=process, args=("p1", 1, ))
p2 = multiprocessing.Process(target=process, args=("p2", 3, ))
p3 = multiprocessing.Process(target=process, args=("p3", 5, ))
p4 = multiprocessing.Process(target=process, args=("p4", 2, ))

#실제 구동은 여기서 시작된다.
p1.start()
p2.start()
p3.start()
p4.start()

cnt = 0
#루프를 10번 돌면서 1초마다 Main THREAD를 출력한다.
while cnt < 10:
    #main thread를 부모 프로세스로 보면 됨.
    print("Main Thread...")
    time.sleep(1)


# In[6]:


import multiprocessing

def withdraw(balance, lock):
    for _ in range(20000):
        lock.acquire() # lock을 획득한다.
        balance.value -= 1
        lock.release() # lock을 해제한다.
        
def deposit(balance, lock):
    for _ in range(20000):
        lock.acquire()
        balance.value += 1
        lock.release()
        
def perform_transaction():
    balance = multiprocessing.Value('i', 20000)
    
    #스핀락을 만들 때 아래와 같이 만든다.
    lock = multiprocessing.Lock()
    
    p1 = multiprocessing.Process(target=withdraw, args=(balance, lock, ))
    p2 = multiprocessing.Process(target=deposit, args=(balance, lock, ))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    print("Final Balance = {}".format(balance.value))
    
for _ in range(10):
    perform_transaction()
    
    
# 여기서 구동한 프로그램의 값이 일정해야 하는데 값이 일정하지 않고 실행할 때 마다 들쭉 날쭉하다.
# 이러한 문제 유발의 원인은 동기화 문제이며 이런 문제의 유형을 Race 


# In[1]:


import time
from multiprocessing import Pool
def f(x):
    return x * x
pool = Pool(processes = 4)
res = pool.apply_async(f, (10, ))
print(pool.map(f, range(10)))
it = pool.imap(f, range(10))
print(it.next())
print(it.next())
print(it.next())


# In[3]:


import time
from multiprocessing import Pool

def f(x):
    return x * x


pool = Pool(processes = 4)

t = time.mktime(time.localtime())

# Async: Asychronous(비동기 처리)를 의미한다.
# 비동기 처리와 동기 처리의 차이는?
# 비동기는 일단 싹 다 질러놓고 상황 대응은 다음
# 동기는 반드시 상호간의 협약이 필요하다.(암묵적이던 명시적이던)

# apply_async도 람다 방식으로 f 에 1000000개를 연산 할수 있도록 준비한다.
# 여기서 만든 개수 / 프로세스 숫자 만큼 처리할 데이터의 개수를 분배하게 된다. 
# 그래서 아래쪽 실제 map을 활용해서 람다 처리를 한다.

res = pool.apply_async(f, (1000000, ))

print(pool.map(f, range(1000000)))

# 모든 처리가 끝났으니 걸린 시간을 t1에 저장한다.
t1 = time.mktime(time.localtime()) - t

t = time.mktime(time.localtime())

# 여기서는 단순히 그냥 for문으로 쭉 돌린다.
# 단순히 for문만 사용하면 병렬처리가 되지 않는다.
for i in range(1000000):
    print(i * i)

t2 = time.mktime(time.localtime()) - t

# 왜 성능차이가 4배 나지 않나요?
# 프로세스를 4개 생성하는데 추가적인 시간이 들어가므로 프로세스 생성시간을 제외한 3.x배 정도의 성능이 최대다.
print("t1 =", t1)
print("t2 =", t2)


# In[5]:



# thread를 사용하기 위한 라이브러리
import threading
# 전역변수 x가 critical section이 된다.
x = 0
lock = multiprocessing.Lock()
# 값은 1씩 증가
def inc_glob():
    global x
    x += 1
    
# 실제 스레드가 하는 일
def taskofThread():
    for _ in range(100000):
        lock.acquire()
        inc_glob()
        lock.release()

def thread_main():
    global x
    x = 0
    
    # 스레드를 생성하고 taskofThread가 돌아가도록 한다.
    # 해당 스레드는 taskofThread를 실행한다.
    t1 = threading.Thread(target = taskofThread)
    t2 = threading.Thread(target = taskofThread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
for i in range(10):
    thread_main()
    print("x = {1} after Iteration {0}.".format(i, x))


# In[6]:


def square(n):
    return n * n

myList = [1,2,3,4,5]
res = []

for num in myList:
    res.append(square(num))

print(res)


# In[10]:


def square(n):
    return n * n
myList = [1, 2, 3, 4, 5]
res = []
for num in myList:
    res.append(square(num))
print(res)

import multiprocessing
import os
def square(n):
    print("Worker Process id for {0}: {1}".format(n, os.getpid()))
    return n * n
myList = [1, 2, 3, 4, 5]
p = multiprocessing.Pool()
res = p.map(square, myList)
print(res)


# In[1]:


import os
import numpy as np
import multiprocessing

dx = 0.000001

def calc(x):
#     print("Process id for {0}: {1}".format(x, os.getpid))
    return dx * (x ** 2)

def integralRange(start, end):
    curX = np.arange(start, end, dx)
#     print(curX)
    p = multiprocessing.Pool()
    res = p.map(calc, curX)
    area = np.sum(res)
    return area

area = integralRange(0, 3)
print("x^2의 0 ~ 3까지 정적분 결과 =", area)


# In[2]:


from PIL import Image
img = Image.open("pepe.jpg")

img.show()


# In[ ]:




