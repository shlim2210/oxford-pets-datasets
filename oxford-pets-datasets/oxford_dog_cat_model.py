import tensorflow as tf
import pandas as pd
import math
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_sequential_model(input_shape):
    model = keras.Sequential(
        [
            layers.Input(input_shape),
            
            # 1st
            # conv2D가 많을수록 합성곱을 많이 하기 때문에 가중치의 수를 많이 줄 수 있음
            # 64개의 필터 / 3X3의 kernel size / strides : 필터가 이동하는 거리 
            # activation : Rectified Linear Unit. (기울기 소실이 없고 기존 활성화 함수에 비해 빠르다는 장점)
            # padding='same': 자동으로 패딩을 삽입해 입력값과 출력값의 크기를 맞춰줌
            # padding='valid': 패딩을 적용하지 않고 필터를 적용해서 출력값의 크기가 작아짐
            layers.Conv2D(64, 3, strides=1, activation='relu', padding='same'),
            layers.Conv2D(64, 3, strides=1, activation='relu', padding='same'),
            # maxpool : 공간의 크기를 점진적으로 줄여 네트워크에서 매개변수와 계산의 양을 줄이고 과적합을 제어.
            # 2 X 2 크기의 필터로 4개의 숫자마다 최대값을 추려냄
            layers.MaxPool2D(),
            # Dropout : 모델의 과적합을 막기 위해 무작위로 특정 노드(입력값)을 0으로 만듦
            layers.Dropout(0.5),
            
            # # 2nd
            # layers.Conv2D(128, 3, strides=1, activation='relu', padding='same'),
            # layers.Conv2D(128, 3, strides=1, activation='relu', padding='same'),
            # layers.MaxPool2D(),
            # layers.Dropout(0.3),
            
            # FC
            layers.GlobalMaxPool2D(),
            # dense layer : hidden layer를 거쳐서 나온 숫자들을 한 곳에 모아주고, 
            # 적절한 함수에 정볼르 전달하기 위한 레이어
            layers.Dense(128, activation="relu"),
            # 출력층의 activation function
                # 1) linear : 디폴트 값, 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력
                # 2) relu : rectifier, 주로 hidden layer에서 많이 쓰임
                # 3) sigmoid : 이진 분류를 할 때 주로 사용
                # 4) softmax : 다중 클래스를 분류 할 때 주로 사용
            layers.Dense(1, activation='sigmoid')
        ]
    )
    return model

input_shape = (256, 256, 3)
model = get_sequential_model(input_shape)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='accuracy'
)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, csv_path, fold, image_size, mode='Train', shuffle=True):
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.fold = fold
        self.image_size = image_size
        self.mode = mode
        self.shuffle = shuffle
        
        self.df = pd.read_csv(csv_path)

        # df['fold']는 검증 데이터의 label(1,2,3,4,5 중 하나). 
        if self.mode == 'train':
            # self.df['fold'] != self.fold : fold가 1이면, 1을 제외한 나머지 행(train)들을 저장
            self.df = self.df[self.df['fold'] != self.fold]
        elif self.mode == 'val':
            # self.df['fold'] == self.fold : fold가 1이면, fold가 1인 행(val)들을 저장
            self.df = self.df[self.df['fold'] == self.fold]
        
        self.on_epoch_end()
    
    # index를 리셋하고 데이터를 섞어서(sample) 가져오는 역할
    def on_epoch_end(self):
        if self.shuffle:
            # frac: 전체 row에서 몇 %의 데이터를 return할 것인지를 설정
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        data = self.df.iloc[start:end]
        #batch_x : image , batch_y : label 
        # get_data()로 이미지의 경로와 label을 각각 batch_x, batch_y에 리스트로 저장하는 역할
        batch_x, batch_y = self.get_data(data)
        return np.array(batch_x), np.array(batch_y)

    def get_data(self, data):
        batch_x = []
        batch_y = []
        
        # 해당 batch의 갯수만큼 반복문 실행. r : 이미지 데이터
        for _, r in data.iterrows():
            file_name = r['filename']
            image = cv2.imread(f'data/images/{file_name}.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image / 255.
            
            label = int(r['species']) - 1
            
            batch_x.append(image)
            batch_y.append(label)
        return batch_x, batch_y

csv_path = 'data/kfolds.csv'

# 학습용 데이터로더 객체
train_generator = DataGenerator(
    batch_size = 9,
    csv_path = csv_path,
    fold = 1,
    image_size = 256,
    mode = 'train',
    shuffle = True
)

# 검증용 데이터로더 객체
valid_generator = DataGenerator(
    batch_size = 128,
    csv_path = csv_path,
    fold = 1,
    image_size = 256,
    mode = 'val',
    shuffle = True
)

# early_stopping : 최적의 값을 찾았을 때 에폭을 정지
# monitor='val_loss' : loss 값이 변할때 마다 / patience = loss값보다 큰 값이 나와도 3번까지 넘김 / mode='min' : loss가 최소가 되는 값을 찾음 / restore_best_weights : True라면 training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원. False라면, 마지막 training이 끝난 후의 weight로 놔둠
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, mode='min', restore_best_weights=False
)

# reduce_on_plateau : 모델의 개선이 없을 경우 learning rate를 수정
# monitor='val_loss' : loss 값이 변할때 마다 / patience=10 : loss값보다 큰 값이 나와도 10번까지 넘김 / factor=0.1 : learing rate에 0.1을 곱해줌 /mode='min' : loss가 최소가 되는 값을 찾음 
reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, mode='min', min_lr=0.0001  
)

# model_checkpoint : 로그를 파일로 남겨주는 함수. 모델을 분석할 때 사용
# save_best_only : 최고값을 경신했을 때만 저장 / save_weights_only=False : 가중치 외에 모델 정보를 저장
filepath = '{epoch:02d}-{val_loss:.2f}.hdf5' 
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min'
)

generator = model.fit(
    train_generator,
    # valdiation_data : loss와 모델의 accuracy를 평가하기 위한 용도로 사용됨.
    validation_data = valid_generator,
    epochs=1,
    callbacks=[
        early_stopping,
        reduce_on_plateau,
        model_checkpoint
    ]
)

# loss, accuracy = model.evaluate(valid_generator)
# print('test loss : ',loss)
# print('accuracy : ',accuracy)