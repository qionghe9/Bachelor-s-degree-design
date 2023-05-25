%%%%%%%%%%%%%%%%%%%%%%%%%after beamform&parcellate->ROI level%%%%%%%%%%%%%%%%%%%%%%

pwd 
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ %更改
osl_startup;

%%%%%%%%%%convert fif to spm
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
sort={'adults','children'};
sort1={'train','test'}

for j=1:length(sort1)
    % datadir=fullfile(basedir,'meg-data','ROI_DATA',sort1{j});
    datadir=fullfile(basedir,'600_data','ROI_DATA',sort1{j});
    for i=1:length(sort)
        fileList=dir(fullfile(datadir,sort{i},'*.fif'));
        fileNames = {fileList.name};
        filefolder={fileList.folder};
        spm_roi_data={};
        fif_name={};
        disp([sort1{j},sort{i},":",length(fileNames)])
        for s=1:length(fileNames)
            fif_name{s}=fullfile(filefolder{s},fileNames{s});
            % [~, name] = fileparts(fif_name{s});
            numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
            spm_path=fullfile(basedir,'600_data','SPM_ROI_DATA',sort1{j},sort{i},numstr);
            % spm_path=fullfile(basedir,'meg-data','SPM_ROI_DATA',sort1{j},sort{i},numstr);
            spm_roi_data{s}=[spm_path '.mat'];
            S2=[];
            S2.outfile = spm_roi_data{s};
            S2.trigger_channel_mask = '0000000000111111';
            osl_import(fif_name{s},S2);
        end
    end
end
 
for j=1:length(sort1)
    % datadir=fullfile(basedir,'meg-data','ROI_DATA',sort1{j});
    datadir=fullfile(basedir,'600_data','ROI_DATA',sort1{j});
    for i=1:length(sort)
        fileList=dir(fullfile(datadir,sort{i},'*.fif'));
        fileNames = {fileList.name};
        filefolder={fileList.folder};
        spm_roi_data={};
        fif_name={};
        disp([sort1{j},sort{i},":",length(fileNames)])
    end
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ %更改
osl_startup;

addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
a={'train','test'};
b={'adults','children'};
data='meg-data';
% data='600_data';

for s1=1:length(a)
    sort1=a{s1};
    datadir=fullfile(basedir,data,'ROI_DATA',sort1);
    for s2=1:length(b)
        sort=b{s2};
        fileList=dir(fullfile(datadir,sort,'*.fif'));
        fileNames = {fileList.name};
        filefolder={fileList.folder};
        spm_roi_data={};
        fif_name={};
        for s=1:length(fileNames)
            fif_name{s}=fullfile(filefolder{s},fileNames{s});
            numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
            spm_path=fullfile(basedir,data,'SPM_ROI_DATA',sort1,sort,numstr);
            spm_roi_data{s}=[spm_path '.mat'];
        end

        %%%%%%%%%setting
        %
        % preproc_name='250Hz/';
        preproc_name='600Hz/';
        s=[];
        S.freq_range=[1 160];
        % S.freq_range=[1 45];
        S.do_prepare=1; 
        S.preproc_name=preproc_name;
        S.num_embeddings=14;
        S.K = 12;

        try preproc_name=S.preproc_name; catch, error('S.preproc_name needed'); end
        try do_prepare=S.do_prepare; catch, do_prepare=1; end
        try num_embeddings=S.num_embeddings; catch, num_embeddings  = 12; end
        try freq_range=S.freq_range; catch, freq_range=[1 160]; end
        % try freq_range=S.freq_range; catch, freq_range=[1 45]; end

        %%%%%%%%%
        settings_prepare=[];
        settings_prepare.sessions_to_do=1:length(spm_roi_data);
        settings_prepare.parcellated_files=spm_roi_data;
        settings_prepare.sort1=sort1;
        settings_prepare.sort2=sort;
        settings_prepare.freq_range=freq_range;
        settings_prepare.parcellation.parcellation_to_use='giles';
        settings_prepare.parcellation.orthogonalisation='symmetric';
        settings_prepare.signflip.num_iters=1500;
        settings_prepare.signflip.num_embeddings=num_embeddings;

        if isfield(S,'signfliptemplatesubj')
            settings_prepare.templatesubj=S.signfliptemplatesubj;
        end
            
        D = spm_eeg_load(spm_roi_data{1});

        if D.fsample < 600
            settings_prepare.do_signflip=1;
            settings_prepare.do_signflip_diagnostics=1;
            settings_prepare.do_hilbert=0;
        else
            settings_prepare.do_signflip=0;
            settings_prepare.do_signflip_diagnostics=0;
            settings_prepare.freq_range = [1 160]; % this a hack to avoid any filtering
            settings_prepare.do_hilbert=1;
        end
        

        if do_prepare
            [~,templatesubj] = prep_parcellated_data( settings_prepare );
            settings_prepare.templatesubj = templatesubj;
        end
    end
end
 
% D = spm_eeg_load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/600_067_1.mat');

basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
data='meg-data';
sort1='test';
% sort='adults';
sort='children';
datadir=fullfile(basedir,data,'ROI_DATA',sort1);

fileList=dir(fullfile(datadir,sort,'*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
spm_roi_data={};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    spm_path=fullfile(basedir,data,'SPM_ROI_DATA',sort1,sort,numstr);
    spm_roi_data{s}=[spm_path '.mat'];
end

%%%%%%%%%setting
%
preproc_name='250Hz/';
% preproc_name='600Hz/';
s=[];
% S.freq_range=[1 160];
S.freq_range=[1 45];
S.do_prepare=1; 
S.preproc_name=preproc_name;
S.num_embeddings=14;
S.K = 12;

try preproc_name=S.preproc_name; catch, error('S.preproc_name needed'); end
try do_prepare=S.do_prepare; catch, do_prepare=1; end
try num_embeddings=S.num_embeddings; catch, num_embeddings  = 12; end
% try freq_range=S.freq_range; catch, freq_range=[1 160]; end
try freq_range=S.freq_range; catch, freq_range=[1 45]; end

%%%%%%%%%
settings_prepare=[];
settings_prepare.sessions_to_do=1:length(spm_roi_data);
settings_prepare.parcellated_files=spm_roi_data;
settings_prepare.sort1=sort1;
settings_prepare.sort2=sort;
settings_prepare.freq_range=freq_range;
settings_prepare.parcellation.parcellation_to_use='giles';
settings_prepare.parcellation.orthogonalisation='symmetric';
settings_prepare.signflip.num_iters=1500;
settings_prepare.signflip.num_embeddings=num_embeddings;

if isfield(S,'signfliptemplatesubj')
    settings_prepare.templatesubj=S.signfliptemplatesubj;
end
    
D = spm_eeg_load(spm_roi_data{1});

if D.fsample < 600
    settings_prepare.do_signflip=1;
    settings_prepare.do_signflip_diagnostics=1;
    settings_prepare.do_hilbert=0;
else
    settings_prepare.do_signflip=0;
    settings_prepare.do_signflip_diagnostics=0;
    settings_prepare.freq_range = [1 160]; % this a hack to avoid any filtering
    settings_prepare.do_hilbert=1;
end

if do_prepare
    [~,templatesubj] = prep_parcellated_data( settings_prepare );
    settings_prepare.templatesubj = templatesubj;
end
