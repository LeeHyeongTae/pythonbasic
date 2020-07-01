from dataclasses import dataclass
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
from nltk import FreqDist
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

@dataclass
class Entity:
    def __init__(self):
        context: str
        fname: str
        target: str

    @property
    def context(self) -> str: return self.context

    @context.setter
    def context(self, context): self.context = context

    @property
    def fname(self) -> str: return self.fname

    @fname.setter
    def fname(self, fname): self.fname = fname

    @property
    def target(self) -> str: return self.target

    @target.setter
    def target(self, target): self.target = target


class Service:
    def __init__(self):
        self.texts = []
        self.tokens = []
        self.okt = []
        self.stopwords = []
        self.freqtxt = []
        # 행벡터의 의미를 가진다. [리스트]

    def extract_token(self, payload):
        print('>>1 corpus 에서 token 추출')
        filename = payload.context + payload.fname
        with open(filename, 'r', encoding='utf-8') as f:
            self.texts = f.read()
        print(f'1단계 결과물 : {self.texts[:300]}')

    def extract_hanguel(self, payload):
        print('>>2 corpus 에서 한글 추출')
        texts = self.texts.replace('\n', ' ')
        tokenizer = re.compile(r'^[ㄱ-힣]')
        # r'[]' []하나는 글자 하나를 뜻한다. ^는 not 과 start 두가지 개념이 있음
        # [^]는 not, ^[]은 start 의미로 표현됨.
        self.texts = tokenizer.sub('', texts)
        # 한글이 아닌 ''로만 처리된 값을 tokenizer에 해당하는 것만 담는다.
        print(f'2단계 결과물 : {self.texts[:300]}')

    def conversion_token(self):
        print('>>3 한글 token 변환.')
        pass

    def compound_noun(self):
        print('>>4 복잡명사화.')
        pass

    def extract_stopword(self):
        print('>>5 노이즈 코퍼스에서 토큰 추출')
        pass

    def filtering_text_with_stopword(self):
        print('>>6 노이즈 필터링 후 시그널 추출')
        pass

    def freqent_text(self):
        print('>>7 시그널 중에서 사용빈도 정렬')
        pass

    def wordcloud(self):
        print('>>8 시각화')
        wc = WordCloud



class Controller:
    def __init__(self):
        pass

    #라벨링을 위해서 사전을 다운로드 받은것.
    def download_dictionary(self):
        nltk.download('all')

    def data_analysis(self):
        entity = Entity()
        service = Service()
        service.extract_token()
        service.extract_hanguel()
        service.conversion_token() # 한글
        service.compound_noun()
        service.extract_stopword() # 필요없는 단어 추출
        service.filtering_text_with_stopword()
        service.freqent_text()
        service.wordcloud()


def print_menu():
    print('0. exit\n')
    print('1. 사전 다운로드\n')
    return input('메뉴선택\n')


app = Controller()
while 1:
    menu = print_menu()
    if menu == '1':
        app.download_dictionary()
    if menu == '2':
        app.data_analysis()
    if menu == '0':
        break