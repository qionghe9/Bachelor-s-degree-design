import os 
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.beamformer import make_lcmv, apply_lcmv_raw
import pickle
import scipy
import re
#%matplotlib qt

os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/')

with open('/GPFS/liuyunzhe_lab_permanent/heqiong/temp/adults_raw1_filter.pkl', 'rb') as f:
    raw_recons = pickle.load(f)
raw_recons =raw_recons[0]
####LCMV beamformer源重建
#计算协方差矩阵
#考虑整合不同类型数据：白化&拟合噪声协方差矩阵
data_cov=mne.compute_raw_covariance(raw_recons,tmin=100)#自动白化原始数据
# data_whiten,data_whiten_cov=mne.cov.compute_whitener()
# whiten_data=np.dot(raw_recons,data_whiten)
noise_cov=mne.compute_raw_covariance(raw_recons,tmax=100)
data_cov.plot(raw_recons.info)
#前向模型与头部文件

subject = ''
spacing = 'oct6'
subjects_dir = mne.datasets.fetch_fsaverage(subjects_dir='/GPFS/liuyunzhe_lab_permanent/heqiong/')
trans="/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/bem/fsaverage-trans.fif"

subjects_dir = mne.datasets.fetch_fsaverage(subjects_dir='E:\\Graduation_design')
trans="E:\\Graduation_design\\fsaverage\\bem\\fsaverage-trans.fif"
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
stc.save('stc.stc')
#可视化visualize
mne.viz.plot_source_estimates(stc, subjects_dir=subjects_dir,
                    subject=subject, initial_time=0.1,
                    hemi='lh', views='lateral',
                    time_unit='s', smoothing_steps=5)
lims = [0.3, 0.45, 0.6]
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir,initial_time=0.087, verbose=True)
stc.plot(mode='glass_brain', clim=dict(kind='value', lims=lims), **kwargs)
stc.plot(clim=dict(kind='value', lims=lims), **kwargs)
stc.data.shape
print(stc)
#3D
filters_vec=make_lcmv(raw_recons.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='vector',
                    weight_norm='unit-noise-gain', rank=None)
stc_vec=apply_lcmv_raw(raw_recons,filters_vec)

stc_vec.save('stc_vec.stc')
#可视化visualize
brain = stc_vec.plot(
    clim=dict(kind='value', lims=lims), hemi='both', size=(600, 600),
    views=['sagittal'],
    # Could do this for a 3-panel figure:
    # view_layout='horizontal', views=['coronal', 'sagittal', 'axial'],
    brain_kwargs=dict(silhouette=True),
    **kwargs)
#用所有三种成分可视化最大体素的活动
peak_vox, _ = stc_vec.get_peak(tmin=0.08, tmax=0.1, vert_as_index=True)
ori_labels = ['x', 'y', 'z']
for ori, label in zip(stc_vec.data[peak_vox, :, :100], ori_labels):
    plt.plot(ori, label='%s component' % label)
plt.legend(loc='lower right')
plt.title('Activity per orientation in the peak voxel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a. u.)')
plt.show()

# 读取左右脑的stc文件
stc_lh = mne.read_source_estimate('lh.stc')
stc_rh = mne.read_source_estimate('rh.stc')
# 合并左右脑的stc对象
stc = mne.SourceEstimate(stc_lh.data + stc_rh.data, [stc_lh.vertices[0], stc_rh.vertices[1]], stc_lh.tmin, stc_lh.tstep)


###parcellate提取ROI
labels = mne.read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)
# 打印标签对象
for label in labels:
    print(label)
labels=[i for i in labels if 'unknown' not in i.name]
# labels_name=[i.name for i in labels]
# print(len(labels_name)==len(set(labels_name)))
label_ts = []
for label in labels:
    label_ts.append(mne.extract_label_time_course(stc,label,src, 'mean'))
# 计算每个 ROI 的时间平均值
label_means = [np.mean(ts, axis=1) for ts in label_ts]
#ROI可视化
for sequence in label_ts:
    xsequence=np.squeeze(sequence)[:100]*10
    plt.plot(xsequence)
plt.title('ROI')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

for sequence in label_ts:
    f, Pxx = scipy.signal.welch(np.squeeze(sequence))
    plt.semilogy(f, Pxx)
plt.title('ROI')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#######################################
#头部文件，源空间模型
subject = ''
spacing = 'ico6'
subjects_dir = mne.datasets.fetch_fsaverage(subjects_dir='/GPFS/liuyunzhe_lab_permanent/heqiong/')
trans="/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/bem/fsaverage-trans.fif"
src = mne.setup_source_space(subject, spacing=spacing,subjects_dir=subjects_dir, add_dist=False)
conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(subject=subject, ico=4,conductivity=conductivity,subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
#标签文件
labels = mne.read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)
labels=[i for i in labels if 'unknown' not in i.name]
labels_name=[i.name for i in labels]
#create_info
numbers=range(1,69)
str_numbers = [f"{num}" for num in numbers]
channel_info = {k: v for k, v in zip(str_numbers, labels_name)}
ch_names = list(channel_info.keys())
# info = mne.create_info(ch_names=ch_names,sfreq=600)
info = mne.create_info(ch_names=ch_names,sfreq=250)

def beamform_parcellate(raw_recons):
    data_cov=mne.compute_raw_covariance(raw_recons,tmin=100)
    noise_cov=mne.compute_raw_covariance(raw_recons,tmax=100)
    fwd = mne.make_forward_solution(raw_recons.info, trans=trans, src=src,
                                    bem=bem, meg=True, eeg=False, n_jobs=1)
    filters=make_lcmv(raw_recons.info, fwd, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)
    stc=apply_lcmv_raw(raw_recons,filters)
    label_ts = []
    for label in labels:
        label_ts.append(mne.extract_label_time_course(stc,label,src, 'mean'))
    label_ts= np.squeeze(label_ts)
    raw_roi = mne.io.RawArray(label_ts, info)
    del data_cov,noise_cov,fwd,filters,stc
    return raw_roi

def get_dig(s):
    digits = re.findall(r'\d+', s)
    result = '_'.join(digits)
    return result

def get_file(data_names_path,sort1,sort2):
    basename=f'/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/ROI_DATA/{sort1}/{sort2}/'
    roi_path=[os.path.join(basename,f'{x}_roi.fif') for x in [get_dig(i) for i in data_names_path]]
    return roi_path

# def get_file(data_names_path,sort1,sort2):
#     basename=f'/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/ROI_DATA/{sort1}/{sort2}/'
#     roi_path=[os.path.join(basename,f'{x}_roi.fif') for x in [get_dig(i) for i in data_names_path]]
#     return roi_path

##########
sort1=['adults','children']
sort2=[1,2]
sort3=['train','test']
for s1 in sort1:
    for s2 in sort2:
        data_names_path,nsample=get_path(s1,s2)
        with open(f'{s1}_raw{s2}_filter.pkl', 'rb') as f:
            raw_recons = pickle.load(f)
        path=get_file(data_names_path,sort3[s2-1],s1)
        if s1=='adults':
            for i in range(len(raw_recons)):
                if i==11:
                    continue
                else:
                    raw_roi=beamform_parcellate(raw_recons[i])
                    raw_roi.save(path[i], overwrite=True)
                    # print(path[i])
                    print(s1,":",i)
            # print(11)
        else:
            for i in range(len(raw_recons)):
                if i in [3,4,8,9]:
                    continue
                else:
                    raw_roi=beamform_parcellate(raw_recons[i])
                    raw_roi.save(path[i], overwrite=True)
                    # print(path[i])
                    print(s1,":",i)

# data_names_path,nsample=get_path('adults',1)
# path=get_file(data_names_path,'train',1)

for s1 in sort1:
    for s2 in sort2:
        data_names_path,nsample=get_path(s1,s2)
        # with open(f'{s1}_raw{s2}_filter.pkl', 'rb') as f:
        #     raw_recons = pickle.load(f)
        path=get_file(data_names_path,sort3[s2-1],s1)
        for i in range(len(path)):
            print(path[i])
###########
with open('children_raw1_filter.pkl', 'rb') as f:
    raw_recons2 = pickle.load(f)
path2=get_file(data_names_path,'train','children')
# for i in range(len(raw_recons2)):
#     raw_roi=beamform_parcellate(raw_recons2[i])
#     raw_roi.save(path2[i], overwrite=True)
#     print(i)

# 3,4,9   test 8

# for i in range(10,16):
#     raw_roi=beamform_parcellate(raw_recons2[i])
#     raw_roi.save(path2[i], overwrite=True)
#     print(i)

for i in range(len(raw_recons2)):
    if i in [3,4,9]:
        continue
    else:
        raw_roi=beamform_parcellate(raw_recons2[i])
        raw_roi.save(path2[i], overwrite=True)
        print(i)
###########
with open('adults_raw2_filter.pkl', 'rb') as f:
    raw_recons1 = pickle.load(f)
path1=get_file(data_names_path,'test','adults')

# for i in range(len(raw_recons1)):
#     raw_roi=beamform_parcellate(raw_recons1[i])
#     raw_roi.save(path1[i], overwrite=True)
#     print(i)
# for i in range(12,15):
#     raw_roi=beamform_parcellate(raw_recons1[i])
#     raw_roi.save(path1[i], overwrite=True)
#     print(i)

for i in range(len(raw_recons1)):
    if i==11:
        continue
    else:
        raw_roi=beamform_parcellate(raw_recons1[i])
        raw_roi.save(path1[i], overwrite=True)
        print(i)

###########
del raw_recons1,path1

with open('children_raw2_filter.pkl', 'rb') as f:
    raw_recons2 = pickle.load(f)
path2=get_file(data_names_path,'test','children')

# for i in range(len(raw_recons2)):
#     raw_roi=beamform_parcellate(raw_recons2[i])
#     raw_roi.save(path2[i], overwrite=True)
#     print(i)

# 3,4,9
path2='/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/ROI_DATA/test/children/236_2_roi.fif'
for i in range(len(raw_recons2)):
    if i==15:
        raw_roi=beamform_parcellate(raw_recons2[i])
        raw_roi.save(path2, overwrite=True)
        # print(path2[i])
        print(i)

# for i in range(len(raw_recons2)):
#     if i in [3,4,9]:
#         continue
#     else:
#         print(i)

############

# for i in ['aparc','aparc.a2005s','aparc.a2009s','aparc_sub','HCPMMP1','HCPMMP1_combined','oasis.chubs','PALS_B12_Brodmann','PALS_B12_Lobes','PALS_B12_OrbitoFrontal','PALS_B12_Visuotopic','Yeo2011_17Networks_N1000','Yeo2011_7Networks_N1000']:
#     labels = mne.read_labels_from_annot(subject=subject, subjects_dir=subjects_dir,parc=i)
#     print(i,":",len(labels))

# subject = '/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/'
# subjects_dir = mne.datasets.fetch_fsaverage(subjects_dir='/GPFS/liuyunzhe_lab_permanent/heqiong/')
# labels = mne.read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)

# 读取左右脑标签文件
# lh_labels = mne.read_labels_from_annot(subject=subject, hemi='lh')
# lh_labels=[i for i in lh_labels if 'unknown' not in i.name]
# rh_labels = mne.read_labels_from_annot(subject=subject, hemi='rh')
# labels = mne.morph_labels(lh_labels + rh_labels, subject_to='/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/')

# import nibabel as nib
# import numpy as np
# import igraph as ig
# import os
# os.chdir('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/')
# # 读取 annot 文件
# annot_path = '/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/label/rh.aparc.annot'
# annot_img = nib.freesurfer.read_annot(annot_path)
# data = annot_img[0]
# affine = annot_img[1]
# # load surface data
# surf_file = '/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/surf/lh.white'
# vertices, triangles = nib.freesurfer.read_geometry(surf_file)
# # create graph
# g = ig.Graph()
# g.add_vertices(len(vertices))
# g.add_edges([(t[0], t[1]) for t in triangles])
# g.add_edges([(t[1], t[2]) for t in triangles])
# g.add_edges([(t[2], t[0]) for t in triangles])
# # calculate adjacency matrix
# adj_matrix = np.array(g.get_adjacency().data)
# label_matrix = np.reshape(data, (adj_matrix.shape[0], -1), order='F')
# nifti_img = nib.Nifti1Image(label_matrix, None)
# # 保存为 Nifti 文件
# nifti_path = os.path.splitext(annot_path)[0] + '.nii.gz'
# nib.save(nifti_img, nifti_path)


# data_cov=mne.compute_raw_covariance(raw_recons,tmin=100)
# noise_cov=mne.compute_raw_covariance(raw_recons,tmax=100)
# fwd = mne.make_forward_solution(raw_recons.info, trans=trans, src=src,
#                                 bem=bem, meg=True, eeg=False, n_jobs=1)
# filters=make_lcmv(raw_recons.info, fwd, data_cov, reg=0.05,
#                     noise_cov=noise_cov, pick_ori='max-power',
#                     weight_norm='unit-noise-gain', rank=None)
# stc=apply_lcmv_raw(raw_recons,filters)