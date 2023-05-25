import os 
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import pickle
import re
#%matplotlib qt

# os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/')  
os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/')  

# path='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/MEG_resting/adults/'
# datanames=os.listdir(path)
# data_names_path=[os.path.join(x,'NeuroData','MEG','rest2_tsss.fif') for x in [os.path.join(path, i) for i in datanames]]
# # raw=mne.io.read_raw_fif(data_names_path[0],allow_maxshield=True,preload=True)
# nsample=len(data_names_path)

def prepro(raw):
    raw_downsampled = raw.copy().resample(sfreq=250)
    raw_filter = raw_downsampled.copy().filter(l_freq=1, h_freq=45) 
    # raw_downsampled = raw.copy().resample(sfreq=600)
    # raw_filter = raw_downsampled.copy().filter(l_freq=1, h_freq=160) 
    ica = ICA(n_components=30, random_state=97)
    ica.fit(raw_filter)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
    print('ecg_indices:',ecg_indices)
    raw_filter.load_data()
    ica.plot_sources(raw_filter,show_scrollbars=True,block=True,title='请选择需要去除的成分')
    ica.exclude =ica.exclude+ecg_indices
    raw_recons=raw_filter.copy()
    raw_recons=ica.apply(raw_recons)
    print('ica.exclude:',ica.exclude)
    # raw_filter.plot(title='ICA处理前')
    # raw_recons.plot(title='ICA处理后')
    # del raw_downsampled,raw_filter,ica
    del raw_filter,ica
    return raw_recons

# def mygenerator(nsample):
#     count=0
#     while True:
#         if(count==nsample):
#             return
#         yield count
#         raw=mne.io.read_raw_fif(data_names_path[count],allow_maxshield=True,preload=True) 
#         new_raw=prepro(raw)
#         raw_recons.append(new_raw)
#         count+=1

# raw_recons=[]
# m=mygenerator(nsample)
# print(next(m))
# print(len(raw_recons))

def get_path(people,sort):
    path=f'/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/MEG_resting/{people}/'
    datanames=os.listdir(path)
    data_names_path=[os.path.join(x,'NeuroData','MEG',f'rest{sort}_tsss.fif') for x in [os.path.join(path, i) for i in datanames]]
    # raw=mne.io.read_raw_fif(data_names_path[0],allow_maxshield=True,preload=True)
    nsample=len(data_names_path)
    return data_names_path,nsample

def write_filter_file(people,sort):
    raw_recons=[]
    for path in data_names_path:
        raw=mne.io.read_raw_fif(path,allow_maxshield=True,preload=True)
        newraw=raw.pick_types(meg='grad')
        new_raw=prepro(newraw)
        raw_recons.append(new_raw)
        print(len(raw_recons))
    with open(f'{people}_raw{sort}_filter.pkl', 'wb') as f:
        pickle.dump(raw_recons, f)

# raw=mne.io.read_raw_fif(data_names_path[11],allow_maxshield=True,preload=True)
# newraw=raw.pick_types(meg='grad')

for i in range(2):
    data_names_path,nsample=get_path('adults',i+1)
    write_filter_file('adults',i+1)
    print('done')

for i in range(2):
    data_names_path,nsample=get_path('children',i+1)
    write_filter_file('children',i+1)
    print('done')



data_names_path,nsample=get_path('adults',1)
write_filter_file('adults',1)
data_names_path,nsample=get_path('adults',2)
write_filter_file('adults',2)

data_names_path,nsample=get_path('children',1)
write_filter_file('children',1)
data_names_path,nsample=get_path('children',2)
write_filter_file('children',2)

with open('children_raw2_filter0.pkl', 'rb') as f:
    raw_recons = pickle.load(f)

with open('children_raw2_filter.pkl', 'wb') as f:
    pickle.dump(raw_recons, f)
    

#############################################################################################################
def get_dig(s):
    digits = re.findall(r'\d+', s)
    result = '_'.join(digits)
    return result

def get_file(data_names_path,sort1,sort2):
    basename=f'/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/raw_filter_data/{sort1}/{sort2}/'
    roi_path=[os.path.join(basename,f'{x}_roi.fif') for x in [get_dig(i) for i in data_names_path]]
    return roi_path

with open('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/adults_raw1_filter.pkl', 'rb') as f:
    raw_recons1 = pickle.load(f)
path1=get_file(data_names_path,'train','adults')
for i in range(len(raw_recons1)):
    raw_recons1[i].save(path1[i], overwrite=True)
    print(i)

with open('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/children_raw1_filter.pkl', 'rb') as f:
    raw_recons2 = pickle.load(f)
path2=get_file(data_names_path,'train','children')
for i in range(len(raw_recons2)):
    raw_recons2[i].save(path2[i], overwrite=True)
    print(i)
##############################################################################################################   
#crop bad time
# raw_recons[4].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# raw_recons[4].set_annotations(annotations)

# craw_recons[4].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[4].set_annotations(annotations)

# craw_recons[5].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[5].set_annotations(annotations)

# craw_recons[7].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[7].set_annotations(annotations)

# craw_recons[9].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[9].set_annotations(annotations)

# craw_recons[10].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[10].set_annotations(annotations)

# craw_recons[11].plot()
# annotations = mne.Annotations(onset=[274, 286], duration=[4, 4], description=['bad', 'bad'])
# craw_recons[11].set_annotations(annotations)

#############
os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/')  
c1=[]
c2=[]

for i in range(len(data_names_path)):
    a=os.path.dirname(data_names_path[i])
    b=get_dig(a)
    c1.append('sub_'+b)
    print(c1[i])

df = pd.concat([pd.DataFrame({'adults': c1}), pd.DataFrame({'children':c2})], axis=1)
df.to_csv('output.csv', index=False)