import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from sklearn.model_selection import KFold
# 학습데이터의 분포를 고르게 해주기 위해 사용
from sklearn.model_selection import StratifiedKFold

#Image CLASS-ID SPECIES BREED ID
#ID: 1:37 Class ids
#SPECIES: 1:Cat 2:Dog
#BREED ID: 1-12:Cat 1-25:Dog
#All images with 1st letter as captial are cat images
#images with small first letter are dog images

#list.txt를 csv로 읽어들여 df에 저장. skiprows=건너 뛸 행의 수(주석처리 된 부분은 스킵) delimeter=데이터를 구분할 기준 : ' ' header=None (필드 제목 없이 사용)
df = pd.read_csv('./data/annotations/list.txt', skiprows=6, delimiter=' ', header=None)
df

df.columns = ['filename', 'id', 'species', 'breed']
df

# value_counts() : 해당 Series(컬럼) 종류의 수를 count. sort_index() : Series를 index에 의해 sort
print(df['species'].value_counts())
print(df['species'].value_counts().sort_index())


# 이미지 파일들이 들어있는 경로
image_dir = 'data\\images\\'
# xml파일들(각 이미지 파일의 정보가 담겨있는(이미지 크기, 얼굴 위치 등))
bbox_dir = 'data\\annotations\\xmls\\'
# 이미지에서 내가 필요한 부분만 다른 색으로 표현하기 위한 ROI가 들어있는 파일
seg_dir = 'data\\annotations\\trimaps\\'

# 이미지파일 이름을 가져와서 리스트로 저장
image_files = glob(image_dir + '*.jpg')
len(image_files)

# bounding box 
bbox_files = glob(bbox_dir + '*.xml')



# 고르게 분포시키기 위해 skf 사용
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

df['fold'] = -1
# df['id]를 기준으로 고르게 분포되도록 나눠줌
# idx : 인덱스 값 / t : train / v : valid / kf.split(df) Kfold로 df를 나눈 객체, 1부터 시작함 / enumerate(kf.split(df), 1) ==> 5번 돌게 됨
for idx, (t, v) in enumerate(skf.split(df, df['id']), 1):
    df.loc[v, 'fold'] = idx

df.to_csv('data/kfolds.csv', index=False)