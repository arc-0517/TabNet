## Library

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings('ignore')

import platform
if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
        plt.rc('font', family='NanumMyeongjo')
plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결


########################################################################################
# Data Load
########################################################################################
data_dir = "./data"
file_name = "master_df.csv"

data = pd.read_csv(os.path.join(data_dir, file_name), encoding='cp949')

# data preprocessing
np.random.seed(2022)
if "Set" not in data.columns:
    data["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(data.shape[0],))

for i in ['train', 'valid', 'test']:
    globals()[f'{i}_indices'] = data[data.Set==i].index

device_list = ['건조기','공청기','냉장고','세탁기','스타일러','식기세척기','에어컨','오븐','워시타워']
device_dict = {w: i for i, w in enumerate(device_list)}
vocab_size = len(device_dict)
print("# of vocab size : ", vocab_size)

# make inputs/targets
inputs, targets = [], []
for seq in data.total_seq:
    inputs.append(np.asarray([device_dict[n] for n in seq.split('-')][:-1]))
    targets.append(np.asarray([device_dict[seq.split('-')[-1]]]))
pad_inputs = np.array([np.pad(v, (0, (data.seq_len.max()-1) - len(v)), 'constant') for v in inputs])

tab_df = data[data.columns[4:-1]]

nunique = tab_df.nunique()
types = tab_df.dtypes

categorical_columns = []
categorical_dims =  {}
for col in tab_df.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, tab_df[col].nunique())
        l_enc = LabelEncoder()
        tab_df[col] = tab_df[col].fillna("VV_likely")
        tab_df[col] = l_enc.fit_transform(tab_df[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        tab_df.fillna(tab_df.loc[train_indices, col].mean(), inplace=True)

unused_feat = ['Set']

features = [ col for col in tab_df.columns if col not in unused_feat]

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

from pytorch_tabnet.tab_network import TabNet
tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":1}


net = TabNet(input_dim=tab_df.shape[1],
             output_dim=6,
             cat_idxs=cat_idxs,
             cat_dims=cat_dims)

