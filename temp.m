%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of parcellation
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));

sort1='train';
% sort='adults';
sort='children';
fileList=dir(fullfile(datadir,sort,'*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
spm_roi_data={};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    spm_path=fullfile(basedir,'meg-data','SPM_ROI_DATA','train',sort,numstr);
    spm_roi_data{s}=[spm_path '.mat'];
end

%%%%%%%%%setting

% Add netlab and fmt
% addpath( fullfile(osldir,'ohba-external','netlab3.3','netlab') );
% addpath( fullfile(osldir,'ohba-external','fmt') );


%
preproc_name='250Hz/';
s=[];
S.freq_range=[1 45];
S.do_prepare=1; 
S.preproc_name=preproc_name;
S.num_embeddings=14;

S.K = 12;


try preproc_name=S.preproc_name; catch, error('S.preproc_name needed'); end
try do_prepare=S.do_prepare; catch, do_prepare=1; end
try num_embeddings=S.num_embeddings; catch, num_embeddings  = 12; end
try freq_range=S.freq_range; catch, freq_range=[1 45]; end

%%%%%%%%%
settings_prepare=[];

settings_prepare.sessions_to_do=1:length(spm_roi_data);
settings_prepare.parcellated_files=spm_roi_data;
settings_prepare.sort1=sort1;
settings_prepare.sort2=sort;
settings_prepare.freq_range=freq_range;

%settings_prepare.parcellation.parcellation_to_use='test';
settings_prepare.parcellation.parcellation_to_use='giles';

%settings_prepare.parcellation.orthogonalisation='innovations_mar';
%settings_prepare.parcellation.innovations_mar_order=14;
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
else
    settings_prepare.do_signflip=0;
    settings_prepare.do_signflip_diagnostics=0;
    settings_prepare.freq_range = [0 0]; % this a hack to avoid any filtering
end
settings_prepare.do_hilbert=0;

tic
if do_prepare
    [~,templatesubj] = prep_parcellated_data( settings_prepare );
    settings_prepare.templatesubj = templatesubj;
end
endtime=toc/60;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of roi_data
pwd %查看
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ %更改
osl_startup;
datadir='/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/';
fif_files{1}=fullfile(datadir,'ROI68.fif');
spm_files{1}=[datadir,'spm_meg1.mat'];
spm_files{2}=[datadir,'spm_meg0.mat'];
fif_files{2}=fullfile(datadir,'raw_filter.fif');

%'/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/ROI_DATA/train/adults/067_1_roi.fif'
%convert from fif to spm
if(~isempty(fif_files))
    S2=[];
    for i=1:length(fif_files), % loops over subjects
        S2.outfile = spm_files{i};
        S2.trigger_channel_mask = '0000000000111111'; % binary mask to use on the trigger channel
        D = osl_import(fif_files{i},S2);
        % report.events(D);
    end
end

% load in the SPM M/EEG object
subnum = 1;
D = spm_eeg_load(spm_files{subnum});
R=spm_eeg_load(spm_files{2});
%view
D = oslview(D);
D
D.ntrials
D.chanlabels 
D.chantype
size(D) %D(channels, samples, trials) 
D.fname %查看文件名
D.fsample
has_montage(D)
D = D.montage('switch',0);

%Hilbert envelope
%D = D.montage('switch',3);
ts = D(:,:,:);
Hen = hilbenv(ts);
figure
plot(D.time,ts');
xlabel('Time (s)')
ylabel('Raw signal')
figure
plot(D.time,Hen');
xlabel('Time (s)')
ylabel('Amplitude envelope value')

figure
imagesc(corr(Hen')+diag(nan(68,1)))
axis square
colorbar
title('Envelope correlation before leakage correction')
figure
imagesc(corr(ts')+diag(nan(68,1)))
axis square
colorbar
set(gca,'CLim',[-1 1])
title('Raw correlation before leakage correction')
%corrrct source leakage
D = ROInets.remove_source_leakage(D,'symmetric');
has_montage(D)
ts_lc = D(:,:,:);

figure
imagesc(corr(ts_lc')+diag(nan(68,1)))
axis square
colorbar
set(gca,'CLim',[-1 1])
title('Raw correlation after leakage correction')

Hen_lc = hilbenv(ts_lc);
figure
imagesc(corr(Hen_lc')+diag(nan(68,1)))
axis square
colorbar
title('Envelope correlation after leakage correction')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of opt-hmm

%%%%%%getdata
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
datadir=fullfile(basedir,'meg-data','ROI_DATA','train');
spmfilesdir='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/';
fileList=dir(fullfile(datadir,'adults','*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    signflipped_files{s}=fullfile(spmfilesdir,strcat('sfold_',numstr,'.mat'));
end

signflipped_files=unique(signflipped_files);
isfile = cell2mat(cellfun(@exist,signflipped_files,repmat({'file'},1,length(signflipped_files)),'uniformoutput',0))>0;

disp(['Number of sessions is ' num2str(length(signflipped_files))]);
if (any(~isfile))
    warning('Invalid signflipped_files');
end
signflipped_files=signflipped_files(isfile);

hmm_input={}
T_input={}
for i=1:length(signflipped_files)
    D = spm_eeg_load(signflipped_files{i});
    hmm_input{i}=D(:,:,:)';
    T_input{i}=size(D,2);
end

%%%%%%set options
options = struct();
options.K = 12;
options.order =  0;
options.covtype = 'full';
options.zeromean = 1;
options.embeddedlags = -7:7; 
options.pca = 68 * 2; 
options.standardise = 1;
options.Fs = 250;
% show progress?
options.verbose = 1;
options.initrep = 1; % to make it quicker - leave by default otherwise
options.initcyc = 1; % to make it quicker - leave by default otherwise
options.cyc = 30; % to make it quicker - leave by default otherwise

% A = [1]; % 定义矩阵A
% n = 3; % 复制次数
% B=repmat(A, [1 n])
% cellA = mat2cell(B, size(A, 1), ones(1, n));


%%%%%%run hmm
D = spm_eeg_load(signflipped_files{2});

%[hmm_tde,Gamma_tde] = hmmmar(num2cell(D(:,:,:), [1 2]),{[size(D,2)]},options);
[hmm_tde,Gamma_tde] = hmmmar(D(:,:,:)',size(D,2),options);
[hmm, Gamma, Xi, vpath, GammaInit, residuals, fehist]=hmmmar(D(:,:,:)',size(D,2),options);
[hmm_tde,Gamma_tde] = hmmmar(hmm_input,T_input,options);

size(D,2)
[Gamma,Xi] = hmmdecode(D(:,:,:)',size(D,2),hmm_tde,0) ;
[viterbipath] = hmmdecode(D(:,:,:)',size(D,2),hmm_tde,1);
size(viterbipath) 
%save(results_file,'hmm_tde','Gamma_tde','-append')

%%%%%%run hmmspectramt

options = struct();
options.fpass = [1 40]; % frequency range we want to look at, in this case between 1 and 40 Hertzs.
options.tapers = [4 7]; % internal multitaper parameter
options.Fs = 250; % sampling frequency in Hertzs
options.win = 10 * options.Fs; % window length, related to the level of detail of the estimation;
options.embeddedlags = -7:7;

%查看每个ROI的每个状态的激活情况
tic
spectra_tde = hmmspectramt(D(:,:,:)',size(D,2),Gamma_tde,options);
toc 
%save(results_file,'spectra_tde','-append')

channels_prim_visual_cortex = [26 27];
plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
%plot_hmmspectra (spectra_tde,[],[],[],[]);


%%%%%%run nnmf
params_fac = struct();
params_fac.Base = 'coh';
params_fac.Method = 'NNMF';
params_fac.Ncomp = 2; % set to a higher value (4) to pull out more detailed frequency modes
%频率分解
tic
[spectral_factors,spectral_profiles] = spectdecompose(spectra_tde,params_fac);
toc
%save(results_file,'spectral_factors','spectral_profiles','-append')

atlasfile='/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/label/lh.aparc.gii';
g=gifti(atlasfile);

p = parcellation(atlasfile); % load the parcellation

net_mean = zeros(68,hmm_tde.train.K);
for k = 1:length(spectra_tde.state)
    net_mean(:,k) =  diag(squeeze(abs(spectral_factors.state(k).psd(1,:,:))));
end
net_mean = zscore(net_mean); % show activations for each state relative to the state average
p.osleyes(net_mean); % call osleyes


%%%%%%%
%随时间状态激活的概率
t = 3001:5001; % some arbitrary time segment
figure
area(t/250,Gamma_tde(t,:),'LineWidth',2);  xlim([t(1)/250 t(end)/250])
xlabel('Time'); ylabel('State probability')
title('TDE-HMM' )

%状态转移概率矩阵
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )

%状态生命周期
lifetimes_tde = getStateLifeTimes (Gamma_tde,size(D,2),hmm_tde.train,[],[],false);
lifetimes_tde = lifetimes_tde(:)'; lifetimes_tde = cell2mat(lifetimes_tde);
figure
hist(lifetimes_tde/250,100); xlim([0 0.5]); ylim([0 10000])
xlabel('Life times'); ylabel('No. of visits')
title('TDE-HMM')
%The TDE-HMM, which has a tendency to focus on slower frequencies, has the longest life times.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spatial_basis_file ='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/parcellations/
                    fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz';
p = parcellation(spatial_basis_file);
p.n_parcels%ROI的数量

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of hmm  
pwd 
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ 
osl_startup;
osldir='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/';
addpath( fullfile(osldir,'ohba-external','netlab3.3','netlab') );
addpath( fullfile(osldir,'ohba-external','fmt') );
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/osl/'));

% general settings:
S=[];
S.do_hmm=1; 
S.do_spectral_estimation=0;
S.preproc_name='250Hz/';
S.session_name='train';
S.sort='adults';
S.freq_range=[1 45];
S.spmfilesdir=fullfile('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/','adults');
S.prep_sessions_to_do=1:2;
S.hmm_sessions_to_do=1:2;
S.num_embeddings=14;
S.K = 12;

%S.parcellations_dir = [osldir, '/parcellations'];

%%%%%%%
signflipped_files={};
% unsignflipped_files={};
spmfilesdir='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/';
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
datadir=fullfile(basedir,'meg-data','ROI_DATA','train');
fileList=dir(fullfile(datadir,'adults','*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    % unsignflipped_files{s}=fullfile(spmfilesdir,strcat(numstr,'.mat'));
    signflipped_files{s}=fullfile(spmfilesdir,strcat('sfold_',numstr,'.mat'));
end

% check signflipped_files exist
%signflipped_files_ext = cellfun(@strcat,signflipped_files,repmat({'.mat'},1,length(signflipped_files)),'uniformoutput',0);
signflipped_files=unique(signflipped_files);
isfile = cell2mat(cellfun(@exist,signflipped_files,repmat({'file'},1,length(signflipped_files)),'uniformoutput',0))>0;
hmm_input_spm_files=signflipped_files(isfile);
disp(['Number of sessions is ' num2str(length(signflipped_files))]);
if (any(~isfile))
    warning('Invalid signflipped_files');
end
disp(['Number of sessions is ' num2str(length(signflipped_files))]);

%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of do_spectral_estimation  all data
datap =[];
for subnum = 1:length(hmm.data_files)
    D = spm_eeg_load((hmm.data_files{subnum}));
    embed.tres=1/D.fsample;
    data = osl_teh_prepare_data(D,normalisation,logtrans,[],embed);
    datap = [datap, data];
end

nsubs=length(hmm.data_files)
hmm_sub = hmm; 

% hmm_sub.statepath = hmm.statepath(subj_inds<=nsubs); 
% hmm_sub.gamma = hmm.gamma(subj_inds<=nsubs,:);


% inds = logical(hmm_sub.statepath == 1);
% datap_in=datap(:,inds);

x = hmm_sub.gamma(:,1);
x2 = permute(repmat(x,[1, size(datap,1)]),[2 1]);
datap_in=datap.*x2;


%%%%%%%%%%%

S=[];
S.parcellated_filenames=hmm.data_files;
S.normalisation='voxelwise';
S.assignment='soft';
S.global_only=false;
S.embed.do=0;
S.embed.rectify=false;

S.netmat_method=@netmat_spectramt;
S.netmat_method_options.fsample=hmm.fsample;
S.netmat_method_options.fband=freq_range;
S.netmat_method_options.type='coh';
S.netmat_method_options.full_type='full';
S.netmat_method_options.var_normalise=false;
S.netmat_method_options.reg=2;
S.netmat_method_options.order=0;

params = struct('Fs',Hz); % Sampling rate
params.fpass = S.fband;  % band of frequency you're interested in 
params.tapers = [4 7]; % taper specification - leave it with default values
%params.tapers = [5 9]; % taper specification - leave it with default values
params.p = 0; % interval of confidence - set to 0 if you don?t wish to compute these
%params.win = 10 * Hz; % multitaper window 
%params.win = 5 * Hz; % multitaper window 
%params.win = 2 * Hz; % multitaper window 
params.win = S.reg * Hz; % multitaper window 

params.to_do = [1 0]; % turn off pdc

if params.win>size(data,1)
    params.win=size(data,1)-1;
end

fitmt=hmmspectramt(data,size(data,1),[],params);    
netmats.spectramt=fitmt.state;
netmats.netmat=permute(netmats.spectramt.(S.type),[2 3 1]);

num_embeddings=size(netmats.netmat,3);
num_nodes=size(netmats.netmat,1);
netmats.netmat_full=speye(num_nodes*num_embeddings);%创建稀疏单位矩阵
for node_ind=1:num_nodes
    from = (node_ind-1)*num_embeddings+1;
    to = from+num_embeddings-1;
    for node_ind2=1:num_nodes
        from2 = (node_ind2-1)*num_embeddings+1;
        to2 = from2+num_embeddings-1;
        netmats.netmat_full(from:to,from2:to2)=sparse(diag(permute(netmats.netmat(node_ind,node_ind2,:),[3 1 2])));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of do_spectral_estimation  each data
S=[];
S.parcellated_filenames=hmm.data_files;
S.normalisation='voxelwise';
S.assignment='hard';
S.global_only=false;
S.embed.do=0;
S.embed.rectify=false;

S.netmat_method=@netmat_spectramt;
S.netmat_method_options.fsample=hmm.fsample;
S.netmat_method_options.fband=freq_range;
S.netmat_method_options.type='coh';
S.netmat_method_options.full_type='full';
S.netmat_method_options.var_normalise=false;
S.netmat_method_options.reg=2;
S.netmat_method_options.order=0;


logtrans=0;
normalisation = S.normalisation;
embed=S.embed;
assignment=S.assignment;

D_fnames=S.parcellated_filenames;    
nsubs=length(D_fnames);


disp(['Computing for subj num ' num2str(1)]);

Dp = spm_eeg_load((D_fnames{1}));

    
%state_netmats{subnum}.parcelAssignments=parcelAssignments;
%state_netmats{subnum}.parcellation=Dp.parcellation;

embed.tres=1/Dp.fsample;
    
% returns data as num_nodes x num_embeddings x ntpts     
datap = osl_teh_prepare_data(Dp,normalisation,logtrans,[],embed);
disp(['size(datap):' int2str(size(datap))])
for  subnum=1:nsubs
    hmm_sub = hmm; 
    hmm_sub.statepath = hmm.statepath(hmm.subj_inds==subnum); 
    hmm_sub.gamma = hmm.gamma(hmm.subj_inds==subnum,:);

    %hmm_sub = rmfield(hmm_sub,'MixingMatrix');    
    disp(['sub:' int2str(subnum)])
    for k = 1:hmm_sub.K

        disp(['Computing for state ' num2str(k)]);
        inds = logical(hmm_sub.statepath == k);
        disp(['sum(inds):' int2str(sum(inds))])
    end
end

    datap_in=datap(:,inds);% * normconstant;
    [state_netmats{subnum}.state{k}]=feval(S.netmat_method,datap_in,S.netmat_method_options);



    % try
    %     [state_netmats{subnum}.state{k}]=feval(S.netmat_method,datap_in,S.netmat_method_options);
    %     state_netmats{subnum}.state{k}.ntpts=size(datap_in,2);
    % catch
    %     warning(['State ' num2str(k) ' is not visited in subject ' num2str(subnum)]);
    %     state_netmats{subnum}.state{k}=[];
    %     state_netmats{subnum}.state{k}.ntpts=0;
    % end
    



% [state_netmats{subnum}.global]=feval(S.netmat_method,datap,S.netmat_method_options);
% state_netmats{subnum}.global.ntpts=size(datap,3);

% state_netmats{subnum}.netmat_method=S.netmat_method;
[hmm_input,T_input,~]=getdata(2,'adults');
Intervals = getStateIntervalTimes (hmm.gamma_tde,hmm.hmm.train); 
Intervals = Intervals(:)'; Intervals = cell2mat(Intervals);
figure
hist(Intervals/250,100); xlim([0 0.5]); ylim([0 10000])
xlabel('Intervals'); ylabel('No. of visits')
title('TDE-HMM')
saveas(gcf,'Intervals.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%try the useful of nnmf
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ 
osl_startup;

osldir='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/';
addpath( fullfile(osldir,'ohba-external','netlab3.3','netlab') );
addpath( fullfile(osldir,'ohba-external','fmt') );
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/osl/'));
% add osl_braingraph:
addpath(genpath([osldir,'/osl-core/old']));

savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/adults/store';
load([savebase '/mydata.mat'])

% Spectral Mode NNMF
% nnmf_outfile = fullfile( savebase, ['embedded_HMM_K',int2str(K),'_nnmf']);

S = [];
S.psds = psd(:,:,:,:,:);
S.maxP=4;
S.maxPcoh=4;
S.do_plots = true;
niterations=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S.do_plots = 0;

% Preallocate for SumSquare of residuls(拟合残差的平方)
ncomps = S.maxP;%4
nsamples = size( S.psds,3 );%90
ss = zeros( niterations, ncomps);

% Specify fit function, a unimodal gaussian
gauss_func = @(x,f) f.a1.*exp(-((x-f.b1)/f.c1).^2);%定义一个高斯分布函数

% Default fit options
options = fitoptions('gauss1'); %返回一个结构体，表示拟合一维高斯函数

% constrain lower and upper bounds
options.Lower = [0,1,0];
options.Upper = [Inf,nsamples,nsamples];

% Main loop
winning_value = Inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
res=[];
    
num_nodes=size(S.psds,4);
nfreqbins=size(S.psds,3);
nsubjects=size(S.psds,1);
NK=size(S.psds,2);%k

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute coherency and auto spectra

coh_comps = zeros(size(S.psds,1),size(S.psds,2),nfreqbins,num_nodes,num_nodes); 
auto_spectra_comps = zeros(size(S.psds,1),size(S.psds,2),nfreqbins,num_nodes); 

for ss = 1:nsubjects
    for kk = 1:NK
        psd=squeeze(S.psds(ss,kk,:,:,:));
        for j=1:num_nodes
            auto_spectra_comps(ss,kk,:,j) = psd(:,j,j);
            for l=1:num_nodes                
                cjl = psd(:,j,l)./sqrt(psd(:,j,j) .* psd(:,l,l));
                coh_comps(ss,kk,:,j,l) = cjl;
            end
        end
    end
end

    maxP=S.maxP;
    
    % concat over states after computing the mean over subjects
    psdtmp=[];
    for kk=1:NK
        psdtmp=cat(2,psdtmp, squeeze(mean(abs(auto_spectra_comps(:,kk,:,:)),1)));  
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a=psdtmp;
k=maxP;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [a b]=nnmf_mww(psdtmp,maxP,'replicates',500,'algorithm','als');
    % [a b]=my_nnmf_mww(psdtmp,maxP,'replicates',500,'algorithm','als');
    [anew bnew]=nnmf_mww(psdtmp,maxPcoh,'replicates',500,'algorithm','als');

    load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/adults/store/res.mat');
    nnmf_res = rmfield(res,'coh');
    savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/adults/store';
    for ii = 1:4
        figure('Position',[100 100*ii 256 256])
        h = area(nnmf_res.nnmf_coh_specs(ii,:));
        h.FaceAlpha = .5;
        h.FaceColor = [.5 .5 .5];
        grid on;axis('tight');
        % xticks(20:20:100);
        % xticklabels({'10', '20', '30', '40', '50'});
        % set(gca,'YTickLabel',[],'FontSize',14);
        % set(gca,'XTick',20:20:100);set(gca,'XTickLabel',[10:10:50]);
        print([savebase '/NNMFMode',int2str(ii)],'-dpng')
    end
    % figure;
    % for pp=1:4  
    %     subplot(121);plot(res.nnmf_psd_specs(pp,:),get_cols(pp),'Linewidth',2);ho;
    %     title('NNMF on PSDs');
    % end
    % for pp=1:4  
    %     subplot(122);plot(res.nnmf_coh_specs(pp,:),get_cols(pp),'Linewidth',2);ho;
    %     title('NNMF on Coherences');
    % end
    % legend(get_cols)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
next_nnmf = teh_spectral_nnmf( S );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

