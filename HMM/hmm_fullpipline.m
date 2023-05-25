cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ 
osl_startup;

osldir='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/';
addpath( fullfile(osldir,'ohba-external','netlab3.3','netlab') );
addpath( fullfile(osldir,'ohba-external','fmt') );
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/osl/'));
addpath(genpath([osldir,'/osl-core/old']));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%setting
session_name{1} = 'train';
Fs_to_run{1} = 600;
spm_roi_datapath{1} = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/';

session_name{2} = 'test';
Fs_to_run{2} = 600;
spm_roi_datapath{2} = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/test/';

%%
session_name{1} = 'train';
Fs_to_run{1} = 250;
spm_roi_datapath{1} = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/';

session_name{2} = 'test';
Fs_to_run{2} = 250;
spm_roi_datapath{2} = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%计算成人最好的hmm模型
% The model inference is sensitive to local minima - therefore we run a
% number of models and keep the one with the lowest free energy:
Nruns = 5;
for i=1:Nruns
    [hmm,hmm_output_file]=do_hmm(1,'adults',0);
    save([hmm_output_file,'/',int2str(i),'th_hmm_result'],'-v7.3','hmm')
    disp([i,'done'])
    clear hmm hmm_output_file;
end
Nruns = 5;
for i=1:Nruns
    [hmm,hmm_output_file]=do_hmm(2,'adults',0);
    save([hmm_output_file,'/',int2str(i),'th_hmm_result'],'-v7.3','hmm')
    disp([i,'done'])
    clear hmm hmm_output_file
end
%save(results_file,'hmm_tde','Gamma_tde','-append')
%load([hmm_output_file,'/',int2str(i),'th_hmm_result'],'hmm')
Nruns = 5;
%% determine model with lowest free energy to keep:
for whichstudy=1:2
    wd=spm_roi_datapath{whichstudy};
    for i=1:Nruns
        hmm = load(fullfile(wd,'adults',[int2str(i),'th_hmm_result.mat']));
        FEcompare(i,whichstudy) = hmm.hmm.fehist(end);
    end
end

% wd=spm_roi_datapath{1};
% hmm = load(fullfile(wd,'adults',[int2str(1),'th_hmm_result.mat']));
% hmm.fehist(end);

[~,bestmodel] = min(sum(FEcompare,2));
save(['/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/','adults_bestmodel.mat'],'FEcompare');

bestmodel=4;
K=12;
%%do spectral estimation
% hmm_output_file='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/adults/';
% do_spectral_estimation(hmm_output_file,bestmodel);
hmm_output_file='/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/adults/';
do_spectral_estimation(hmm_output_file,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%计算儿童最好的hmm模型
Nruns = 5;
for i=1:Nruns
    [hmm,hmm_output_file]=do_hmm(1,'children',0);
    save([hmm_output_file,'/',int2str(i),'th_hmm_result'],'-v7.3','hmm')
    disp([i,'done'])
    clear hmm hmm_output_file;
end

Nruns = 5;
for i=1:Nruns
    [hmm,hmm_output_file]=do_hmm(2,'children',0);
    save([hmm_output_file,'/',int2str(i),'th_hmm_result'],'-v7.3','hmm')
    disp([i,'done'])
    clear hmm hmm_output_file
end
%save(results_file,'hmm_tde','Gamma_tde','-append')
%load([hmm_output_file,'/',int2str(i),'th_hmm_result'],'hmm')

%% determine model with lowest free energy to keep:
for whichstudy=1:2
    wd=spm_roi_datapath{whichstudy};
    for i=1:Nruns
        hmm = load(fullfile(wd,'children',[int2str(i),'th_hmm_result.mat']));
        FEcompare(i,whichstudy) = hmm.hmm.fehist(end);
    end
end

% wd=spm_roi_datapath{1};
% hmm = load(fullfile(wd,'adults',[int2str(1),'th_hmm_result.mat']));
% hmm.fehist(end);

[~,bestmodel] = min(sum(FEcompare,2));
save(['/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/','children_bestmodel.mat'],'FEcompare');

clear hmm;

K=12;
%%do spectral estimation
hmm_output_file='/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/children/';
do_spectral_estimation(hmm_output_file,1);

% hmm_output_file='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/children/';
% do_spectral_estimation(hmm_output_file,1);
% hmm_output_file='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/children/';
% do_spectral_estimation(hmm_output_file,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NeuronFig4Analyses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%else plot

% wd=spm_roi_datapath{2};
wd=spm_roi_datapath{1};
sort='adults';
load(fullfile(wd,sort,[int2str(1),'th_hmm_result.mat']));
hmm_tde=hmm.hmm;
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
load(fullfile(wd,sort,[int2str(2),'th_hmm_result.mat']));
hmm_tde=hmm.hmm;
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
load(fullfile(wd,sort,[int2str(3),'th_hmm_result.mat']));
hmm_tde=hmm.hmm;
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
load(fullfile(wd,sort,[int2str(4),'th_hmm_result.mat']));
hmm_tde=hmm.hmm;
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
load(fullfile(wd,sort,[int2str(5),'th_hmm_result.mat']));
hmm_tde=hmm.hmm;
figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
 
% [hmm_input,T_input,~]=getdata(2,'adults');
% resultfiles=fullfile(wd,'adults','else_plot');
% else_plot(resultfiles,hmm_input,T_input);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';
load([savebase '/mydata.mat'])

S = [];
S.psds = psd(:,:,:,:,:);% 8 12 90 68 68
S.maxP=4;
S.maxPcoh=4;


num_nodes=size(S.psds,4);
nfreqbins=size(S.psds,3);
nsubjects=size(S.psds,1);
NK=size(S.psds,2);%k

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

% plot sanity check 查看67\68ROI的psd\coh
ii=61;jj=60;
figure;
subplot(221);plot(squeeze(abs(auto_spectra_comps(1,:,:,ii)))');ho;
subplot(223);plot(squeeze(abs(coh_comps(1,:,:,ii,jj)))');ho;
subplot(224);plot(squeeze(abs(auto_spectra_comps(1,:,:,jj)))');ho;
subplot(221);plot(squeeze(abs(auto_spectra_comps(1,7,:,ii)))', 'LineWidth',3);ho;
subplot(223);plot(squeeze(abs(coh_comps(1,7,:,ii,jj)))', 'LineWidth',3);ho;
subplot(224);plot(squeeze(abs(auto_spectra_comps(1,7,:,jj)))', 'LineWidth',3);ho;
legend(num2str((1:NK)'));
print([savebase '/' int2str(ii) 'VS' int2str(jj)],'-dpng');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/5th_hmm_result.mat')
[hmm_input,T_input,hmm_input_files]=getdata(1,'children')

 load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/5th_hmm_result.mat')
 [hmm_input,T_input,hmm_input_files]=getdata(1,'adults')

X=cell2mat(hmm_input');
T=cell2mat(T_input);
Gamma_tde=hmm.gamma_tde;

options = struct();
options.fpass = [1 45]; 
options.tapers = [4 7]; 
options.Fs = 250; 
options.win = 10 * options.Fs; 
options.embeddedlags = -7:7;

tic
spectra_tde = hmmspectramt(X,T,Gamma_tde,options);
toc/60 %8min
% save(results_file,'spectra_tde','-append')

savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/store';
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';
% select the channels that correspond to primary visual cortex
channels_prim_visual_cortex = [59,57,61];
plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
print([savebase,'/59,57,61'],'-dpng')
 
channels_prim_visual_cortex = [59,15,57,37,61,17];
plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
print([savebase,'/59,15,57,37,61,17'],'-dpng')
 
% channels_prim_visual_cortex = [5 11 15 17 31 37 55 57 59 61 65 67];
% plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
% print([savebase,'/[5,11,15,17,31,37,55,57,59,61,65,67]'],'-dpng')

channels_prim_visual_cortex = [23,49,57];
plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
print([savebase,'/23,49,57'],'-dpng')
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hmm_tde=hmm.hmm;
% lifetimes_tde = getStateLifeTimes (Gamma_tde,T,hmm_tde.train,[],[],false);
% % We concatenate the lifetimes across states
% lifetimes_tde = lifetimes_tde(:)'; lifetimes_tde = cell2mat(lifetimes_tde);

% figure
% hist(lifetimes_tde/250,100); xlim([0 0.5]); ylim([0 17000])
% xlabel('Life times'); ylabel('No. of visits')
% title('TDE-HMM')
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [hmm_input,T_input,hmm_input_files]=getdata(1,'adults');
[hmm_input,T_input,hmm_input_files]=getdata(1,'children');
% ind=logical([1 0 0 1 1 1 1 1 0 1 1 1 0 1]);
ind=logical([1 1 0 1 1 1 1 0 1 0 1 0 0]);
hmm_input_files=hmm_input_files(ind);
for i=1:length(hmm_input_files)

    X=hmm_input(i);
    T=T_input(i);

    options = struct();
    % We go back to 6 states
    options.K = 12;
    options.order =  0;
    options.covtype = 'full';
    options.zeromean = 1;
    options.embeddedlags = -7:7; % 15 lags are used from -7 to 7, this defines the length of the modelled autocorrelations
    options.pca = 68 * 2; % twice the number of regions (see the logic of this on Vidaurre et al. 2018)
    options.standardise = 1;
    options.Fs = 600;
    options.verbose = 1;
    options.initrep = 1; % to make it quicker - leave by default otherwise
    options.initcyc = 1; % to make it quicker - leave by default otherwise
    options.cyc = 30; % to make it quicker - leave by default otherwise

    tic
    [hmm_tde,Gamma_tde] = hmmmar(X,T,options);
    toc/60

    % [viterbipath] = hmmdecode(X,T,hmm_tde,1);

    T=cell2mat(T);

    lifetimes_tde = getStateLifeTimes (Gamma_tde,T,hmm_tde.train,[],[],false);
    % We concatenate the lifetimes across states
    lifetimes_tde = lifetimes_tde(:)'; lifetimes_tde = cell2mat(lifetimes_tde);

    figure
    hist(lifetimes_tde/250,100); xlim([0 0.5]); ylim([0 2000])
    xlabel('Life times'); ylabel('No. of visits')
    title('TDE-HMM')
    % print(['/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/',int2str(i)],'-dpng')
    print(['/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/',int2str(i)],'-dpng')
     
end