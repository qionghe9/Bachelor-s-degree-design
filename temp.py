import os 
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.beamformer import make_lcmv, apply_lcmv_raw
import pickle
#%matplotlib qt

os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/')  
path='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/MEG_resting/adults/'

######导入数据
datanames=os.listdir(path)
data_names_path=[os.path.join(x,'NeuroData','MEG','rest2_tsss.fif') for x in [os.path.join(path, i) for i in datanames]]
raw=mne.io.read_raw_fif(data_names_path[0],allow_maxshield=True,preload=True)

######预处理
#降采样
raw_downsampled = raw.copy().resample(sfreq=250)
#滤波
raw_filter = raw_downsampled.copy().filter(l_freq=1, h_freq=45)  
#ICA
ica = ICA(n_components=30, random_state=97)# 找出与ECG模式匹配的 IC
ica.fit(raw_filter)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
print('ecg_indices:',ecg_indices)
raw_filter.load_data()
ica.plot_sources(raw_filter,show_scrollbars=True,block=True,title='请选择需要去除的成分')
ica.exclude =ica.exclude+ecg_indices
raw_recons=raw_filter.copy()
raw_recons=ica.apply(raw_recons)
print('ica.exclude:',ica.exclude)
del raw_downsampled,raw_filter,ica

######LCMV beamformer源重建
#计算协方差矩阵
data_cov=mne.compute_raw_covariance(raw_recons,tmin=100)
noise_cov=mne.compute_raw_covariance(raw_recons,tmax=100)
#前向模型与头部文件
subject = ''
spacing = 'ico6'
subjects_dir = mne.datasets.fetch_fsaverage(subjects_dir='/GPFS/liuyunzhe_lab_permanent/heqiong/')
trans="/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/bem/fsaverage-trans.fif"
src = mne.setup_source_space(subject, spacing=spacing,
                        subjects_dir=subjects_dir, add_dist=False) # 创建标准脑的源空间
conductivity = (0.3, 0.006, 0.3)  # 头皮、颅骨和脑脊液的电导率
# inner_skull = mne.bem.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.085)
model = mne.make_bem_model(subject=subject, ico=4,conductivity=conductivity,subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(raw_recons.info, trans=trans, src=src,
                            bem=bem, meg=True, eeg=False, n_jobs=1) # 创建基于标准脑的正向模型
#空间滤波器
#2D
filters=make_lcmv(raw_recons.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=None)
stc=apply_lcmv_raw(raw_recons,filters)

#3D矢量图
filters_vec=make_lcmv(raw_recons.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='vector',
                    weight_norm='unit-noise-gain', rank=None)
stc_vec=apply_lcmv_raw(raw_recons,filters_vec)


######parcellate提取ROI
labels = mne.read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)
for label in labels:
    print(label)
labels=[i for i in labels if 'unknown' not in i.name]
for label in labels:
    print(label)

label_ts = []
for label in labels:
    label_ts.append(mne.extract_label_time_course(stc,label,src, 'mean'))
# 计算每个 ROI 的时间平均值
label_means = [np.mean(ts, axis=1) for ts in label_ts]

del data_cov,noise_cov,stc,fwd,bem,filters,src,model

with open('my_list.pkl', 'wb') as f:
    pickle.dump(label_ts, f)

with open('adults_raw1_filter.pkl', 'rb') as f:
    raw_recons = pickle.load(f)
raw_recons =raw_recons[0]
# data, times = raw_recons.get_data(return_times=True)

with open('my_list.pkl', 'rb') as f:
    label_ts = pickle.load(f)
label_ts= np.squeeze(label_ts)

labels_name=[i.name for i in labels]
numbers=range(1,69)
str_numbers = [f"{num}" for num in numbers]
channel_info = {k: v for k, v in zip(str_numbers, labels_name)}
ch_names = list(channel_info.keys())
info = mne.create_info(ch_names=ch_names,sfreq=250)
raw_sim = mne.io.RawArray(label_ts, info)
# raw_sim.info=raw_recons.info
raw_sim.save('ROI68.fif', overwrite=True) #是否覆盖原文件
raw_recons.save('raw_filter.fif', overwrite=True)

df = pd.DataFrame.from_dict(channel_info,orient='index', columns=['label_name'])
df.to_csv('channel_info.csv')



# import sys
# def fibonacci(n): # 生成器函数 - 斐波那契
#     a, b, counter = 0, 1, 0
#     while True:
#         if (counter > n): 
#             return
#         yield 
#         a, b = b, a + b
#         counter += 1
# f = fibonacci(4) # f 是一个迭代器，由生成器返回生成
# while True:
#     try:
#         print (next(f))
#     except StopIteration:
#         sys.exit()

