#requests,csv 사용전 import해준다
import requests
import csv
import re
import json
#beautiful_soup사용전 import해준다
from bs4 import BeautifulSoup as bs
#csv를 사용하여 만들 파일명을 정한다.
csv_filename="car6.csv"
#csv 객체를 만들어 내용을 저장하는 옵션을 정해준다.
csv_open = open(csv_filename,"w+",encoding='utf-8')
csv_writer = csv.writer(csv_open)
#제일 첫줄에 목차의 개념으로 한 줄 작성한다.
csv_writer.writerow(('가격','이미지','제조사','차종','트림','년식','주행거리'))
## 파일 읽기
# 읽기(r) 모드로 Hello.txt 파일을 file 객체로 열기
file = open('car.txt', 'r', encoding='UTF8')
# file 객체에서 문자열을 읽은 후 변수 x에 저장
data = file.read()
soup = bs(data, 'html.parser')
price=[]
image=[]
brand=[] #제조사
name =[]#차종
trim=[]#트림
year=[]
km=[]
price_list = soup.select('td>strong')
for item in price_list:
    price.append(item.text)
img_list = soup.select('tr>td')
for item in img_list:
    img = item.find_all('img')
    image.append(img_src)
brand_list = soup.select('span.cls>strong')
for item in brand_list:
    brand.append(item.text)
name_list = soup.select('span.cls>em')
for item in name_list:
    name.append(item.text)
trim_list = soup.select('span.dtl>strong')
for item in trim_list:
    trim.append(item.text)
year_list = soup.select('span.yer')
for item in year_list:
    year.append(item.text)
km_list = soup.select('span.km')
for item in km_list:
    km.append(item.text)
for i,t in enumerate(price):
    csv_writer.writerow((price[i],image[i],brand[i],name[i],trim[i],year[i],km[i]))
#csv를 종료한다.
csv_open.close()