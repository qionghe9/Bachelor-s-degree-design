import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from scipy import signal
import pickle
#%matplotlib qt

# os.getcwd()  #获取当前工作目录
# os.chdir()   #修改当前工作目录

###导入数据
# data_path=sample.data_path('E:\\Graduation_design\\sample')
basedir='E:\\Graduation_design\\meg_data'
path='E:\\Graduation_design\\meg_data\\adults'
datanames=os.listdir(path)
data_names_path=[os.path.join(x,'NeuroData','MEG','rest2_tsss.fif') for x in [os.path.join(path, i) for i in datanames]]#批量获取文件名

raw=mne.io.read_raw_fif(data_names_path[0],allow_maxshield=True,preload=True)
######预处理
#基线校正
# new_raw= raw.copy().apply_baseline((None, 0))
# raw.plot(title='基线校正前')
# new_raw.plot(title='基线校正后')
##降采样
raw_downsampled = raw.copy().resample(sfreq=250)
#滤波
raw_filter = raw_downsampled.copy().filter(l_freq=1, h_freq=45)  
#ICA
ica = ICA(n_components=30, random_state=97)
ica.fit(raw_filter)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
print('ecg_indices:',ecg_indices)
raw_filter.load_data()
ica.plot_sources(raw_filter,show_scrollbars=True,block=True,title='请选择需要去除的成分')
print(ica)
ica.exclude =ica.exclude+ecg_indices
print(ica)
raw_recons=raw_filter.copy()
raw_recons=ica.apply(raw_recons)
raw_filter.plot(title='ICA处理前')
raw_recons.plot(title='ICA处理后')
plt.show(block=True)

ica.plot_scores(ecg_scores,title='ICA component "ECG match"') # ICA component "ECG match"的分数条形图
ica.plot_sources(raw) # 绘制应用于原始数据的 IC，突出显示 ECG 匹配
ica.plot_components()
ica.plot_properties(raw, picks=ica.exclude)#诊断信息图




