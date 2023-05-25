function [hmm_input,T_input,hmm_input_files]=getdata(n,sort,h)
session_name{1} = 'train';
Fs_to_run{1} = 250;
spm_roi_datapath{1} = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/';

session_name{2} = 'test';
Fs_to_run{2} = 250;
spm_roi_datapath{2} = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/';

    % for iFreq=1:length(Fs_to_run{n})
    %         Fs=Fs_to_run{n}(iFreq);
    %         if Fs==600
    %             preproc_name='600Hz/';
    %             freq_range=[0 0];
    %         else
    %             preproc_name='250Hz/';
    %             freq_range=[1 45];
    %         end
freq_range=[1 45];
% signflipped_files={};
enveloped_files={};
unsignflipped_files={};
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
datadir=fullfile(basedir,'meg-data','ROI_DATA',session_name{n});
spmfilesdir=fullfile('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/',session_name{n},sort);
fileList=dir(fullfile(datadir,sort,'*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    signflipped_files{s}=fullfile(spmfilesdir,strcat('sfold_',numstr,'.mat'));
    % unsignflipped_files{s}=fullfile(spmfilesdir,strcat(numstr,'.mat'));
    enveloped_files{s}=fullfile(spmfilesdir,strcat('h',numstr,'.mat'));
end
% signflipped_files=unique(signflipped_files);
% isfile = cell2mat(cellfun(@exist,signflipped_files,repmat({'file'},1,length(signflipped_files)),'uniformoutput',0))>0;        
% disp(['Number of sessions is ' num2str(length(signflipped_files))]);
% if (any(~isfile))
%     warning('Invalid signflipped_files');
% end
% signflipped_files=signflipped_files(isfile);

hmm_input={};
T_input={};
if h
    for i=1:length(enveloped_files)
        D = spm_eeg_load(enveloped_files{i});
        hmm_input{i}=D(:,:,:)';
        T_input{i}=size(D,2);
    end
    hmm_input_files=enveloped_files;
    disp(['Number of sessions is ' num2str(length(enveloped_files))]);
else
    for i=1:length(signflipped_files)
        D = spm_eeg_load(signflipped_files{i});
        hmm_input{i}=D(:,:,:)';
        T_input{i}=size(D,2);
    end
    hmm_input_files=signflipped_files;
    disp(['Number of sessions is ' num2str(length(signflipped_files))]);
end   