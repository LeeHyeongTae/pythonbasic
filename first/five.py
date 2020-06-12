#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image

img = Image.open("pepe.jpg")

#아래 명시한 위치들을 기반으로 크롭해서 나온다.
#(0, 0)이 좌측 상단임을 잊지 말자.
dim = (0, 0, 400, 400)
crop_img = img.crop(dim)


crop_img.show()


# In[2]:


from PIL import Image

img = Image.open("pepe.jpg")

# image에는 color space 라는 공간이 있는데, 이 공간에서 컬러 값들을 빼고 조도로만 이미지를 재구성하면 그레이스케일이 된다.
# 조도로만 재구성하면 좋은것이 컴퓨터가 색상에 민감하게 반응하지 않게 된다.
grayscale = img.convert("L")
grayscale.show()


# In[4]:


from PIL import Image

img = Image.open("pepe.jpg")

# 이미지를 리사이즈 할때는 반드시 아래와 같이 타입을 튜플 형태로 넣어줘야 한다.
resized_img = img.resize((400,400))
resized_img.show()


# In[6]:


from PIL import Image
from PIL import ImageEnhance
img = Image.open("pepe.jpg")

# ImageEnhance로 밝기를 조절해서 사진상에 나와있는 잡티등을 없애주게 만들 수 있다.
enhanced_img = ImageEnhance.Brightness(img)
#숫자가 높을 수록 강렬하게 빛을 밝혀 잡티를 없애준다. 그러므로 위의 크롭과 함께써서 부분적으로 없애자
enhanced_img.enhance(1).show()


# In[10]:


from PIL import Image

img = Image.open("pepe.jpg")

# 반시계 방향으로 회전시킨다.
# 내부에는 radian 표현이 아닌 degree(각도) 표현을 사용한다.
rotated_img = image.rotate(90)
rotated_img.show()


# In[12]:


from PIL import Image
from PIL import ImageEnhance
img = Image.open("pepe.jpg")

# 콘트라스트 조절 대조 강화
contrasted_img = ImageEnhance.Contrast(img)
# 숫자가 높을 수록 대조 강화
contrasted_img.enhance(3).show()


# In[17]:


from skimage import io

img = io.imread('pepe.jpg')
io.imshow(img)


# In[18]:


from skimage import io

# 이미지 읽어오기(형태는 행렬 형태임)
img = io.imread('pepe.jpg')
# 이름을 new로 저장하는 작업
io.imsave('new_pepe.jpg', img)
# 다시불러와 저장되었는지 확인
img = io.imread('new_pepe.jpg')
io.imshow(img)


# In[20]:


from skimage import data, io

# 글자 인식(OCR)에 활용하는 예제중 하나.
io.imshow(data.text())
io.show()


# In[21]:


from skimage import color, io

img = io.imread('pepe.jpg')
# 위의 필로우에서 사용한 컨버터 L과 통일하다.
gray = color.rgb2gray(img)
io.imshow(gray)
io.show()


# In[26]:


from PIL import Image
from PIL import ImageFilter

img = Image.open('pepe.jpg')
# 가우시안 블러는 차량에서는 잡음제거용으로 사용되며 실시간 영상에서는 특정인물의 모자이크로 활용된다.
blur_img = img.filter(ImageFilter.GaussianBlur(5))
blur_img.show()


# In[23]:


from skimage import io
from skimage import filters

img = io.imread('pepe.jpg')
# 가우시안 통계함수가 라플라시안 적분을 기반으로 산출된다.
# 그래서 sigma 값이 별도로 존재하는데 이 값이 높으면 높을 수록 분산이 커지기 때문에 숫자가 크면 클수록 모자이크가 강회된다.
out = filters.gaussian(img, sigma = 5)
io.imshow(out)
io.show()


# In[27]:


from skimage import io
from skimage.morphology import disk
from skimage import color
from skimage import filters

img = io.imread('pepe.jpg')
img = color.rgb2gray(img)
out = filters.median()
io.imshow(out)
io.show()


# In[30]:


from PIL import Image
from PIL import ImageFilter

img = Image.open('pepe.jpg')
# 그레이스케일 작업
img = img.convert("L")
# 전용 필터를 만들기 위한 커스텀 연산 커널
new_img = img.filter(
    ImageFilter.kernel(
        # 3 3 행렬의 연산 커널
        # 연산 대상은 [1,2,3]
        #[4,5,6]
        #[7,8,9]
        # 위 행렬이 이미지 행렬과 컨버트 연산을 수행하게 된다 결국 미분이 진행된다.
        # 첫번째 인자는 행렬의 차원, 두번째 인자는 해당 행렬에 배치된 값들
        (3, 3), [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
)

# 필터 이론에 대해 조금만 설명하자면 철수가 A지점에 있다. 철수가 B지점을 가려고 한다 사이의 거리는 10m 갔다오는데 100분이 걸렸다.
# 철수의 이동속도는 ?
# 결국 컴퓨터가 리미트를 표현 할수 없기 때문에 미분또한 평균으로 접근하게 된다. 즉 단순히 삼각형의 기울기 구하기 문제가 된다.
new_img.show()


# In[1]:


import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from skimage.io import imread
from skimage.color import rgb2gray
img = imread('pepe.jpg')
img = rgb2gray(img)
# 임계치를 잡아오는 함수
thresh_value = threshold_otsu(img)
# 지정한 임게치보다 작은 값을 흰색이나 ~
thresh_img = img > thresh_value

# 영역을 지정해서 반복적으로 패턴을 검색
block_size = 35
adaptive_img = threshold_local(thresh_img, block_size, offset = 10)
fig, axes = plt.subplots(nrows = 3, figsize = (20, 10))
ax0, ax1, ax2 = axes

plt.gray()

ax0.imshow(img)
ax0.set_title('Origin')

ax1.imshow(thresh_img)
ax1.set_title('Global Thresholding')

ax2.imshow(adaptive_img)
ax2.set_title('Adaptive Thresholding')


# In[3]:


import matplotlib.pyplot as plt
from skimage.transform import(hough_line, probabilistic_hough_line)
from skimage.feature import canny
from skimage import io, color
img = io.imread('highway.jpg')
img = color.rgb2gray(img)
edges = canny(img, 3)
io.imshow(edges)
io.show()
lines = probabilistic_hough_line(
    edges, threshold = 10, line_length = 5, line_gap = 3
)
fig, axes = plt.subplots(
    1, 3, figsize = (15, 5), sharex = True, sharey = True
)
ax = axes.ravel()
ax[0].imshow(img, cmap = plt.cm.gray)
ax[0].set_title('Origin')
ax[1].imshow(edges, cmap = plt.cm.gray)
ax[1].set_title('Canny Edge')
ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot(
        (p0[0], p1[0]), (p0[1], p1[1])
    )
ax[2].set_xlim(0, img.shape[1])
ax[2].set_ylim(img.shape[0], 0)
ax[2].set_title('Probabilistic Hough')
for a in ax:
    a.set_axis_off()
plt.tight_layout()
plt.show()


# In[4]:


from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
mnist = datasets.load_digits()
imgs = mnist.images
data_size = len(imgs)
io.imshow(imgs[3])
io.show()
# Image 전처리
imgs = imgs.reshape(len(imgs), -1)
labels = mnist.target
# 로지스틱 회귀 분석 준비
LR_classifier = LogisticRegression(
    C = 0.01, penalty = 'l2', tol = 0.01
)
# 3/4 는 학습에 활용, 1/4은 평가용으로 활용
LR_classifier.fit(
    imgs[:int((data_size / 4) * 3)],
    labels[:int((data_size / 4) * 3)]
)
# 평가 진행
predictions = LR_classifier.predict((imgs[int((data_size / 4)):]))
target = labels[int((data_size / 4)):]
# 성능 측정
print("Performance Report: \n%s\n" %
     (metrics.classification_report(target, predictions))
)


# In[5]:


from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage import io, color, feature, transform
mnist = datasets.load_digits()
imgs = mnist.images
data_size = len(imgs)
# Image 전처리
imgs = imgs.reshape(len(imgs), -1)
labels = mnist.target
# 로지스틱 회귀 분석 준비
LR_classifier = LogisticRegression(
    C = 0.01, penalty = 'l2', tol = 0.01, max_iter = 1000000000
)
# 3/4 는 학습에 활용, 1/4은 평가용으로 활용
LR_classifier.fit(
    imgs[:int((data_size / 4) * 3)],
    labels[:int((data_size / 4) * 3)]
)
# 사용자가 지정한 이미지를 넣어서
# 실제로 이미지의 숫자를 판별하는지 검사해보도록 한다.
digit_img = io.imread('digit.jpg')
digit_img = color.rgb2gray(digit_img)
# MNIST 사용시 주의할점: 이미지 크기를 28 x 28 보다 작게 맞춰야함
digit_img = transform.resize(digit_img, (8, 8), mode="wrap")
digit_edge = feature.canny(digit_img, sigma = 1)
io.imshow(digit_edge)
# 딥러닝 하는 프로세스, 마지막에 무조건 한번 flastten을 해줘야 한다. 자료구조 = 그래프 이론
digit_edge = [digit_edge.flatten()]
# 평가 진행
predictions = LR_classifier.predict(digit_edge)
print(predictions)


# In[9]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pepe.jpg')
# OpenCV가 처리하는 Color Space 방식과 처리방식이 달라
# color space를 서로 맞게 다시 컨버팅해줘야 함.
# cv2 cvtColor는 컨버트칼라의 약자.
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[1]:


import cv2, numpy as np

cv2.namedWindow('Test')

fill_val = np.array([255, 255, 255], np.uint8)

def trackbar_callback(idx, val):
    fill_val[idx] = val
    
cv2. createTrackbar('R', 'Test', 255, 255, lambda v: trackbar_callback(2,v))

cv2. createTrackbar('G', 'Test', 255, 255, lambda v: trackbar_callback(2,v))

cv2. createTrackbar('B', 'Test', 255, 255, lambda v: trackbar_callback(2,v))

while True:
    img = np.full((500, 500, 3), fill_val)
    cv2.imshow('Test', img)
    key = cv2.waitKey(3)
    
    if key == 27:
        break
        
cv2.destoryAllWindows()


# In[1]:


import cv2

cam = cv2.VideoCapture(0)
while(cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()


# In[7]:


import cv2
import numpy as np
# Region of Interest(관심영역)
def roi(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
fps = 30
title = 'normal speed video'
delay = int(1000/fps)
cam = cv2.VideoCapture("challenge.mp4")
while(cam.isOpened()):
    ret, frame = cam.read()
    if ret != True:
        break
    # 영상 프레임을 가져오면
    # 해당 영상의 높이값과 폭을 얻을 수 있다.
    height = frame.shape[0]
    width = frame.shape[1]
    # 우리가 관심을 가지려고 하는 영역을 지정(삼각형)
    region_of_interest_vertices = [
        (0,height),
        (width / 2, height / 2),
        (width, height)
    ]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    edges = cv2.Canny(gray, 235, 243, 3)
    cropped_img = roi(
        edges,
        np.array(
            [region_of_interest_vertices], np.int32
        )
    )
    cv2.imshow('frame',cropped_img)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


# In[ ]:




