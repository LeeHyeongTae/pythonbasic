#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("3rd programming")


# In[2]:


# java나 다른 언어의 중괄호 표기를 ':'이 대체하고 있다.
class PyTestClass:
    """it's for Test Class"""
    pass

# dir() 사용할수 있는 모듈(라이브러리)에 어떤 것들이 있는지 보여준다.
print(dir())
# type은 당연히 class로 나온다.
print(type(PyTestClass))


# In[6]:


#자바에서는 메서드만 따로 빠져있는 경우가 없었음.
# 외부에 def A()만 딸랑 있는 경우
# 동일하게 메서드라고 생각하면 된다.
# 메서드를 호출해야 하듯 마찬가지로 호출을 해줘야 동작이 된다.

class Person:
    Name = "Default Name"
    
    def Print(self):
        print("My name is {0}".format(self.Name))
        
p1 = Person()
p1.Print()
print("##########")

# 현재는 전부 public이므로 외부에서 마음대로 참조 및 변경이 가능
p1.Name = "Python3"
p1.Print()
#직접 클래스 에 있는 프린트를 호출한다는 방식으로도 사용이 가능하다.
# 이것은 순수한 인터프리터 언어라 가능한 것임.
Person.Print(p1)

p1 = Person()
p1.Name = "Python3"

p2 = Person()

print("#########")
#클래스 구조는 같아도 객체가 서로 다르므로 내부에 설정하는 값은 다르다.
print("p1.Name =", p1.Name)
print("p2.Name =", p2.Name)


# In[8]:


# 클래스 내부 변수를 아래와 같이 언제든지 마음대로 추가 할 수 있다.
Person.title = "Python3 Test"
print("p1.title =", p1.title)
print("p2.title =", p2.title)
print("Person.title =", Person.title)

p1.age = 20
print("p1.age =", p1.age)


# In[10]:


#str1 전역으로 생성되어 있음.
str1 = "Not CLass Member"

class NonSelfTest:
    str1 = ""
    
    def Set(self, msg):
        self.str1 = msg
        
    def Print(self):
        print(self.str1)
        print(str1)
        
# 객체를 생성
test = NonSelfTest()
# Setter를 활용해서 "Test Message"를 설정
test.Set("Test Message")
# 결과를 출력해서 메세지를 보고 싶었으나 출력된 것은 전역 변수인 "Not Class Member"가 출력됨.
# self를 명시하지 않고 이름이 같은게 있다면 전역변수에 접근하게 된다.
test.Print()


# In[14]:


class ClassTest:
    data = "Default"
    
ct1 = ClassTest()
ct2 = ClassTest()

print(ct1.data)
print(ct1.data)

# __class__.data로 지정하면 해당 객체의 클래스 타입을 파악하여 해당 클래스에 대한 모든 객체의 값을 변경한다.
ct1.__class__.data = "Change"
print("########")
print(ct1.data)
print(ct2.data)

#__class__가 없으면 대상 객체만 변경된다.
ct2.data = "Only this"
print('##########')
print(ct1.data)
print(ct2.data)


# In[19]:


class Vehicle:
    pass
class Fish:
    pass
# 파이썬에서 상속은 아래와 같이 한다. Airplane은 Vehicle을 상속 받았음을 의미한다.
class Airplane(Vehicle):
    pass

v, a = Vehicle(), Airplane()

# 자식은 부모에 대한 인스턴스가 될 수 있다.
print("v is instance of Vehicle:", isinstance(v, Vehicle))
print("a is loV:", isinstance(a, Vehicle))
print("v is instance of Airplane:", isinstance(v, Airplane))
print("v is Instance of Object:", isinstance(v, object))
print("v is instance of Fish:", isinstance(v, Fish))
print("int is instance of Object:", isinstance(int, object))


# In[21]:


class Vehicles:
    #__init__ 이 키워드 붙어있으면 Constructor
    #init: 초기화
    def __init__(self, value):
        self.Value = value
        print("Vehicle Class Constructor ! Value = ", value)
    #__del__ 이 키워드가 붙으면 소멸자
    def __del__(self):
        print("Vehicles Class Destructor!")

def test():
    #t라는 객체는 test 함수 내에서만 살아있는 지역변수(Stack)
    t = Vehicles(333)
    #그러므로 test 함수가 끝나면 소멸자가 호출된다.
    print("정말?")
    
test()
print("레알?")


# In[26]:


class CntManager:
    cnt = 0
    
    def __init__(self):
        CntManager.cnt += 1
        
    # self를 사용하지 않고 클래스 자체의 cnt로 접근하라고 하고 있음.
    def staticPrintCnt():
        print("Instance cnt :", CntManager.cnt)
    # staticmethod - 언제든지 외부에서 쉽게 참조하여 호출할 수 있게 만들어줌 (일종의 별칭)   
    # 즉 클래스에 있는 메서드를 조금더 이름이 단순한 sPrintCnt로 대체
    sPrintCnt = staticmethod(staticPrintCnt)
    
    # 결국 classmethod로 만든 classPrintCnt의 cls는 self와 동일한 역할을 수행하게 된다.
    def classPrintCnt(cls):
        print("Instace cnt : ", cls.cnt)
        
    # classmethod - 디폴트로 첫번째 인자가 자신의 클래스 셋팅됨.    
    cPrintCnt = classmethod(classPrintCnt)

# 결국 생성자를 통해 모든 CntManager 기반의 객체들의 cnt 값을 증가시킴
a, b, c = CntManager(), CntManager(), CntManager()

CntManager.sPrintCnt()
b.sPrintCnt()

#파라미터 표기가 없어도 알아서 디폴트로 클래스 자신이 셋팅됨.
CntManager.sPrintCnt()
c.sPrintCnt()


# In[27]:


# 자식이 부모 클래스를 상속받을 때 생성자를 어떻게 처리해야 하는가 ?
class Person:
    #Person의 생성자
    def __init__(self, name, phoneNum):
        self.name = name
        self.phoneNum = phoneNum
        
    def printInfo(self):
        print("Info(name:{0}, Phone Num: {1})".fomat(self.name, self.phoneNum))
        
    def printPersonData(self):
        print("Person(name: {0}, Phone Num: {1})".format(self.name, self.phoneNum))
        
class Student(Person):
    #Student의 생성자
    #자바와는 다르게 생성자 내에서 부모 클래스에 대한 부분을 처리해줘야 한다.
    def __init__(self, name, phoneNum, subject, studentId):
        self.name = name
        self.phoneNum = phoneNum
        self.subject = subject
        self.studentId = studentId
        
p = Person("Maron", "010-0000-0000")
s = Student("Luna", "010-1234-5678", "Electric", "1000001")

# 생성된 객체를 __dict__ 통해서 살펴보면 내부의 정보들을 획득할 수 있다.
print(p.__dict__)
print(s.__dict__)


# In[28]:


import math

class Circle(object):
    def __init__(self, r):
        self.r = r
        
    def setter(self, r):
        self.r = r

    # pi * r^2 (원의 넓이)
    @property
    def area(self):
        return math.pi * self.r ** 2
    
    # 2 * pi * r (원의 둘레)
    @property
    def circumference(self):
        return 2 * math.pi * self.r

c = Circle(3.0)
print(c.area)
print(c.circumference)

c.setter(7.0)
print(c.area)
print(c.circumference)


# In[32]:


class DummyClass:
    def __init__(self):
        self.var1 = 3
        self._var2 = 'python'
        self.__var3 = 'Class'
        
dp = DummyClass()

# 노 언더바 - public 
print(dp.var1)
# 언더바 1개 - protected
print(dp._var2)
# 언더바 2개 - private
# print(dp.__var3)
print(dp._DummyClass__var3)


# In[34]:


class Animal:
    def __init__(self, name, height, weight):
        self.name = name
        self.height = height
        self.weight = weight
    
    def info(self):
        print("name:", str(self.name))
        print("height:", str(self.height))
        print("weight:", str(self.weight))
        
class Carnivore(Animal):
    def __init__(self, name, height, weight, feed, sound):
        # Animal 에 있는 생성자를 호출 강제 요청 super와 동일하다.
        Animal.__init__(self, name, height, weight)
        self.feed = feed
        self.sound = sound
        
    def sounds(self):
        print(str(self.name)+" : "+ str(self.sound))

wolf = Carnivore("Timber Wolf", 140, 75, "Meat", "Howl")

print(wolf.__dict__)
wolf.sounds()


# In[37]:


class Parent(object):
    def __init__(self, num):
        self.num = num
    
    def printMsg(self):
        print("Im Super Class")
        
class Child(Parent):
    def __init__(self, num):
        #super(Child, self)라는 것이 Child의 super Class를 불러오라는 의미
        # Super Class를 불러와서 그것의 __init__이므로 결국 Super Class의 생성자를 호출하는 것과 같다.
        super(Child, self).__init__(num)
        
    def printMsg(self):
        print("im a sub class : [%s]" % str(self.num))
        super(Child, self).printMsg()
        
c = Child(3)
c.printMsg()


# In[ ]:




