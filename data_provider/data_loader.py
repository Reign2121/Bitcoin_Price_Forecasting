import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset): #파이토치 Dataset 상속
    
    #생성자. 객체가 생성될 때 호출되고, 여러 인자를 받아 데이터셋을 초기화하는 역할
    def __init__(self, size=None, flag = 'train',
                 features=None, data_path = None, root_path = None,
                 target=None, scale=True, timeenc=0, freq='h', train_only=False ):    #기본적으로 지정되는 생성자의 인자들
        
        # size [seq_len, label_len, pred_len]
        
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]    
            
        #전달한 인자들
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.train_only = train_only
        
        self.__read_data__() #데이터 읽어들이고 전처리하는 역할을 담당한다. 밑에서 선언함

    def __read_data__(self): #데이터 읽어들이고 전처리를 어떻게 할 지 지정해주는 함수
        self.scaler = StandardScaler() #표준화
        df_raw = pd.read_csv(os.path.join(self.root_path + self.data_path))

        '''
        df_raw.columns: ['Time', ...(other features), target feature] 
        
        '''
        cols = list(df_raw.columns) #raw 데이터의 컬럼들을 리스트에 담음
        cols.remove(self.target) #타겟 제거
        cols.remove('Time') #날짜 제거
        df_raw = df_raw[['Time'] + cols + [self.target]] #raw dataset 순서에 따라 정렬 (업데이트)
        # print(cols)

        #데이터 분할
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1)) # 70%
        num_test = int(len(df_raw) * 0.2) # 20%
        num_vali = len(df_raw) - num_train - num_test #10%
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] #각 데이터 셋의 시작 인덱스 [train: 0, valid: 학습데이터에서 seq_len만큼 뺀, test: 전체 데이터에서 테스트 만큼 빼고 seq_len만큼 뺌]
        border2s = [num_train, num_train + num_vali, len(df_raw)] #각 데이터 셋의 끝 인덱스
        border1 = border1s[self.set_type] #self.set_type == [0 (train), 1 (test), 2(val)] / self.set_type에 따라 적절한 시작 인덱스를 선택함
        border2 = border2s[self.set_type] #self.set_type == [0 (train), 1 (test), 2(val)] / self.set_type에 따라 적절한 끝 인덱스를 선택함

        if self.features == 'M' or self.features == 'MS': #만약 다변량이면,
            cols_data = df_raw.columns[1:] #학습에 쓰일 데이터 컬럼을 date를 빼고 다 지정함
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]] #학습에 쓰일 데이터 컬럼을 target으로만 지정함

        #스케일링 #표준화
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) #train 데이터로 스케일러 학습
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values) #학습에 쓸 데이터에 스케일러 적용
        else:
            data = df_data.values #스케일링 X

        #Timestamp encoding #트랜스포머 전용
        df_stamp = df_raw[['Time']][border1:border2]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        
        if self.timeenc == 0: #그냥 일반적인 time_encoding 
            df_stamp['month'] = df_stamp.Time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Time'], axis=1).values
        elif self.timeenc == 1: #커스텀 함수 이용
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq) 
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2] # 참고 border1 = border1s[self.set_type] * self.set_type == [0 (train), 1 (test), 2(val)] / self.set_type에 따라 적절한 시작 인덱스를 선택함
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    #데이터 반환하는 함수
    def __getitem__(self, index):
        s_begin = index #시퀀스 시작은 인덱스
        s_end = s_begin + self.seq_len #시퀀스 끝은 seq_len까지
        
        # label_len은 디코더를 쓰는 트랜스포머 모델들에 이용된다.(트랜스포머 전용) 
        r_begin = s_end - self.label_len #레이블은 시퀀스 끝에서 label_len을 뺀 지점부터 
        r_end = r_begin + self.label_len + self.pred_len #레이블 끝은 pred_len의 끝까지

        seq_x = self.data_x[s_begin:s_end] #인풋 시퀀스 (인풋 시리즈 길이)
        seq_y = self.data_y[r_begin:r_end] #아웃풋 시퀀스 (예측 시리즈 길이)
        seq_x_mark = self.data_stamp[s_begin:s_end] #타임 스탬프
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 #전체 길이

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) #역표준화, 원래 스케일로 돌림
    
    
    
###################################################################################################################################
    
    
    
class Dataset_Pred(Dataset):
    def __init__(self, size=None, flag = 'train',
                 features=None, data_path = None, root_path = None,
                 target=None, scale=True, inverse=False, timeenc=0, freq='h', cols = None , train_only=False):# size [seq_len, label_len, pred_len]
        
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['Time', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('Time')
        df_raw = df_raw[['Time'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len  #이 보더가 trainset과 testset의 경계를 나눔
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['Time']][border1:border2]
        tmp_stamp['Time'] = pd.to_datetime(tmp_stamp.Time)
        pred_dates = pd.date_range(tmp_stamp.Time.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['Time'])
        df_stamp.Time = list(tmp_stamp.Time.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Time.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['Time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
