%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 250HZ plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ 
osl_startup;

osldir='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/';
addpath( fullfile(osldir,'ohba-external','netlab3.3','netlab') );
addpath( fullfile(osldir,'ohba-external','fmt') );
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/data-analysis/'));
addpath(genpath('/GPFS/liuyunzhe_lab_permanent/heqiong/osl/'));
% add osl_braingraph:
addpath(genpath([osldir,'/osl-core/old']));

%% Replay Paper Figure 4: State spectral profiles

% Define sample rate
sample_rate = 600;

%load hmm
load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/adults/1th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/test/children/4th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/2th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/1th_hmm_result.mat')


load('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/test/adults/3th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/test/children/1th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/adults/4th_hmm_result.mat')
load('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/children/1th_hmm_result.mat')

%load/compute new_state_ordering
disttoplot = plotMDS_states(hmm.hmm);
[~,new_state_ordering] = sort(disttoplot(:,1));%升序排列

% if any(new_state_ordering(2:end) ~= new_state_ordering(1:end-1)+1)
%     hmm = hmm_permutestates(hmm,new_state_ordering);
%     hmmfile = [hmmdir,'hmm',template_string,'_parc_giles_symmetric__pcdim80_voxelwise_embed14_K',int2str(K),'_big1_dyn_modelhmm.mat'];
%     save(hmmfile,'new_state_ordering','-append');
%     disttoplot = disttoplot(new_state_ordering,:);
% end

% hmm_new = hmm_permutestates(hmm.hmm,new_state_ordering);

% % [~,hmmT,~]=getdata(2,'adults');
% [~,hmmT,~]=getdata(1,'adults');
% scan_T = cell2mat(hmmT);%

%save path
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/store';
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';

savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/adults/store';
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/children/store';
if ~exist(savebase)
    mkdir(savebase);
end

% parc_file = ['fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz'];
% parc = parcellation(parc_file);
% statelabels={'1','2','3','4','5','6','7','8','9','10','11','12'};

% %% SECTION 1: STATE SPECIFIC FREQUENCY BREAKDOWN:

% % first we must check that the soft computed frequency spectra have been computed, and if not, compute them:

% %load state_netmats_mtsess soft
% mtfilename = [savebase '/state_netmats_mtsess_2_vn0_soft_global0.mat'];

% if ~exist(mtfilename)
%     error('Soft state timecourses not found - rerun this analysis!');
% end

%%%%
[psd,coh] = loadMTspect(savebase,0,new_state_ordering);% 14 12 90 68 68/13 12 90 68 68;12    12   543    68    68

% check for bad subjects:
BadSubj = any(isnan(psd(:,:) + coh(:,:)),2);%返回badsubj所在行 badsub缺少RSN状态的sub
psd(BadSubj,:,:,:,:) = [];% 8 12 90 68 68/6;
coh(BadSubj,:,:,:,:) = [];

save([savebase '/mydata.mat'],'psd','coh','-v7.3')

savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/test/adults/store';
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/test/children/store';
load([savebase '/mydata.mat'])

nparcels = 68;%ROI的数量
K=12;
% colorscheme = set1_cols();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot psd and coh as function of freq averaged over parcels:
clear f_band,offdiags,psd_all,coh_all,psd_cross
f_band=1:size(psd,3);
offdiags = eye(nparcels)==0;
psd_all = zeros(K,length(f_band));
coh_all= zeros(K,length(f_band));
psd_cross= zeros(K,length(f_band));

for kk=1:K
    G = squeeze( mean(abs(coh(:,kk,f_band,offdiags)),4)); %14   543
    P = squeeze(mean(abs(psd(:,kk,f_band,~offdiags)),4));
    P_off=squeeze(mean(abs(psd(:,kk,f_band,offdiags)),4));
    coh_all(kk,:) = mean(G,1);
    coh_ste(kk,:) = std(G,[],1)./sqrt(length(G));
    psd_all(kk,:) = mean(P,1);
    psd_ste(kk,:) = std(P,[],1)./sqrt(length(P));
    % psd_cross(kk,:) = mean(P_off,1);
    % psd_c_ste(kk,:) = std(P_off,[],1)./sqrt(length(P_off));
end
colorscheme = set1_cols();
%
psd_all=psd_all(:,255:end);
psd_ste=psd_ste(:,255:end);
f_band=1:size(psd_all,2);
%%psd
figure('Position',[440 508 708 290]);
for k=1:K
    if mod(k,2)==1
         ls{k} = '-';
     else
         ls{k}='--';
    end
    shadedErrorBar(0.5*f_band,psd_all(k,:),psd_ste(k,:),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
    statelabels{k} = ['State ',int2str(k)];
    h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
end
grid on;
title('PSD per state');
% plot4paper('Frequency');%X\Y轴加上标签,输入参数小于2,Y轴标签为空
xlabel('Frequency');
for k=1:K,h(k).DisplayName=['State ',int2str(k)];end
leg=legend(h,'Location','EastOutside');

print([savebase '/CrossStatePSD_long'],'-dpng')
 
% D=spm_eeg_load('/GPFS/liuyunzhe_lab_permanent/heqiong/600_data/SPM_ROI_DATA/train/adults/600_067_1.mat')

%%coh
figure('Position',[440 508 708 290]);
for k=1:K;
    if mod(k,2)==1
         ls{k} = '-';
     else
         ls{k}='--';
    end
    shadedErrorBar(0.5*f_band,coh_all(k,:),coh_ste(k,:),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
    statelabels{k} = ['State ',int2str(k)];%
    h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
end
grid on;
title('Coherence per state');
% plot4paper('Frequency');
xlabel('Frequency');
for k=1:K,h(k).DisplayName=['State ',int2str(k)];end
leg=legend(h,'Location','EastOutside');

print([savebase '/CrossStateCOH_long'],'-dpng')
 
myplot(savebase,psd,coh,[59,57,61])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Infer NNMF spectral decomposition:
    
% Spectral Mode NNMF
nnmf_outfile = fullfile( savebase, ['embedded_HMM_K',int2str(K),'_nnmf']);
    
% Compute the NNMF
if ~isfile([nnmf_outfile,'.mat'])

    S = [];
    S.psds = psd(:,:,:,:,:);
    S.maxP=4;
    S.maxPcoh=4;
    S.do_plots = true;
    nnmf_res  = run_nnmf( S, 10 );
    nnmf_res = rmfield(nnmf_res,'coh');

    save(nnmf_outfile,'nnmf_res')
     

    % Visualise the mode shapes
    for ii = 1:4
        figure('Position',[100 100*ii 256 256])
        h = area(nnmf_res.nnmf_coh_specs(ii,:));
        h.FaceAlpha = .5;
        h.FaceColor = [.5 .5 .5];
        grid on;axis('tight');
        xticks(20:20:100);
        xticklabels({'10', '20', '30', '40', '50'});
        % set(gca,'YTickLabel',[],'FontSize',14);
        % set(gca,'XTick',20:20:100);set(gca,'XTickLabel',[10:10:50]);
        print([savebase '/NNMFMode',int2str(ii)],'-dpng')
    end
else
    load( nnmf_outfile );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Infer NNMF_WB spectral decomposition:
% Spectral Mode NNMF_WB
nnmf_outfile = fullfile( savebase, ['embedded_HMM_K',int2str(K),'_nnmfWB']);
    
% Compute the NNMF_WB
if ~isfile([nnmf_outfile,'.mat'])

    S = [];
    S.psds = psd(:,:,:,:,:);
    S.maxP=2;
    S.maxPcoh=2;
    S.do_plots = true;
    nnmf_res  = run_nnmf( S, 10 );
    nnmf_res = rmfield(nnmf_res,'coh');

    save(nnmf_outfile,'nnmf_res')

    % Visualise the mode shapes
    for ii = 1:2
         figure('Position',[100 100*ii 256 256])
       h = area(nnmf_res.nnmf_coh_specs(ii,:));
        h.FaceAlpha = .5;
        h.FaceColor = [.5 .5 .5];
        grid on;axis('tight');
        xticks(20:20:100);
        xticklabels({'10', '20', '30', '40', '50'});
        % set(gca,'YTickLabel',[],'FontSize',14);
        % set(gca,'XTick',20:20:100);set(gca,'XTickLabel',[10:10:50]);
        print([savebase '/NNMF_WBMode',int2str(ii)],'-dpng')
    end
else
    load( nnmf_outfile );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% PSD and Coherence scatter plots:
I = logical(eye(nparcels));
for pp=1:3
    figure('Position',[702 470 453 320]);
    for kk=1:K
        scatter_coh(:,kk) = abs(squeeze(sum(nnmf_res.nnmf_coh_maps(kk,pp,:,:),4)))/nparcels;
        scatter_psd(:,kk) = abs(squeeze(nnmf_res.nnmf_psd_maps(kk,pp,:)));
        scatter(scatter_psd(:,kk),scatter_coh(:,kk),20,'MarkerFaceColor',colorscheme{kk});hold on;
        scatterplotlabels{kk} = ['RSN-State ',int2str(kk)];
    end
    xlim([0,0.1]);ylim([0,0.004]);
    xlabel('PSD');
    ylabel('Coherence');
    % plot4paper('PSD','Coherence');
    box on;axis square;
    legend(scatterplotlabels,'Location','EastOutside');
    Ylabel = get(gca,'YTick');
    for i=1:length(Ylabel);Ylabelstr{i} = num2str(Ylabel(i));end
    set(gca,'YTickLabel',Ylabelstr);
    print([savebase,'/ScatterPlot_mode',int2str(pp)],'-dpng');
    %close all;

    pval_anova_nb_coh(pp) = anova1(scatter_coh);
    pval_anova_nb_psd(pp) = anova1(scatter_psd);
    for kk=1:K
        [~,pval_ttest_nb_coh(pp,kk)] = ttest(scatter_coh(:,kk),mean(scatter_coh(:,setdiff([1:K],kk)),2),'Tail','right');
        [~,pval_ttest_nb_psd(pp,kk)] = ttest(scatter_psd(:,kk),mean(scatter_psd(:,setdiff([1:K],kk)),2),'Tail','right');
    end
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nparcels = 68;%ROI的数量
K=12;
colorscheme = set1_cols();
% and same for wideband:
nnmf_outfileWB = fullfile( hmmdir, ['embedded_HMM_K',int2str(K),template_string,'_nnmfWB']);

load( nnmf_outfileWB );
nnmfWB_res=nnmf_res;
%figure('Position',[440 519 408 279]);
figure('Position',[702 470 453 320]);
for kk=1:K
    scatter_coh(:,kk) = abs(squeeze(sum(nnmfWB_res.nnmf_coh_maps(kk,1,:,:),4)))/nparcels;
    scatter_psd(:,kk) = abs(squeeze(nnmfWB_res.nnmf_psd_maps(kk,1,:)));
    scatter(scatter_psd(:,kk),scatter_coh(:,kk),20,'MarkerFaceColor',colorscheme{kk});hold on;
    scatterplotlabels{kk} = ['RSN-State ',int2str(kk)];

end
xlabel('PSD');
ylabel('Coherence');
xlim([0,0.1]);ylim([0,0.004]);
box on;
legend(scatterplotlabels,'Location','EastOutside');
Ylabel = get(gca,'YTick');
for i=1:length(Ylabel);Ylabelstr{i} = num2str(Ylabel(i));end
set(gca,'YTickLabel',Ylabelstr);
% grid on;
axis square;
print([savebase,'/ScatterPlot_wideband'],'-dpng');
%close all;

% compute stats:

pval_anova_wb_coh= anova1(scatter_coh);
% print([savebase,'/boxplot_coh'],'-dpng')
pval_anova_wb_psd = anova1(scatter_psd);
% print([savebase,'/boxplot_psd'],'-dpng')
for kk=1:K
    [~,pval_ttest_wb_coh(kk)] = ttest2(scatter_coh(:,kk),squash(scatter_coh(:,setdiff([1:K],kk))));
    [~,pval_ttest_wb_psd(kk)] = ttest2(scatter_psd(:,kk),squash(scatter_psd(:,setdiff([1:K],kk))));
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%差异图绘制
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';
load([savebase '/mydata.mat'])

nparcels = 68;%ROI的数量
K=12;
colorscheme = set1_cols();

f_band=1:size(psd,3);
offdiags = eye(nparcels)==0;
psd_all = zeros(length(f_band));
coh_all= zeros(length(f_band));
psd_cross= zeros(length(f_band));

G = squeeze( mean(abs(coh(:,3,f_band,offdiags)),4));
P = squeeze(mean(abs(psd(:,3,f_band,~offdiags)),4));
P_off=squeeze(mean(abs(psd(:,3,f_band,offdiags)),4));
coh_allc= mean(G,1);
coh_stec= std(G,[],1)./sqrt(length(G));
psd_allc= mean(P,1);
psd_stec= std(P,[],1)./sqrt(length(P));
psd_crossc = mean(P_off,1);
psd_c_stec = std(P_off,[],1)./sqrt(length(P_off));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/store';
load([savebase '/mydata.mat'])

nparcels = 68;%ROI的数量
K=12;
colorscheme = set1_cols();

f_band=1:size(psd,3);
offdiags = eye(nparcels)==0;
psd_all = zeros(length(f_band));
coh_all= zeros(length(f_band));
psd_cross= zeros(length(f_band));

G = squeeze( mean(abs(coh(:,2,f_band,offdiags)),4));
P = squeeze(mean(abs(psd(:,2,f_band,~offdiags)),4));
P_off=squeeze(mean(abs(psd(:,2,f_band,offdiags)),4));
coh_alla= mean(G,1);
coh_stea= std(G,[],1)./sqrt(length(G));
psd_alla= mean(P,1);
psd_stea= std(P,[],1)./sqrt(length(P));
psd_crossa = mean(P_off,1);
psd_c_stea = std(P_off,[],1)./sqrt(length(P_off));

psd_all=[psd_allc;psd_alla];
psd_ste=[psd_stec;psd_stea];
coh_all=[coh_allc;coh_alla];
coh_ste=[coh_stec;coh_stea];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sort={'children','adults'};
%%psd
figure('Position',[440 508 708 290]);
subplot(2,1,1)
for k=1:2
    if mod(k,2)==1
         ls{k} = '-';
     else
         ls{k}='--';
    end
    % shadedErrorBar(0.5*f_band,psd_all(k,:),psd_ste(k,:));hold on;
    shadedErrorBar(0.5*f_band,psd_all(k,:),psd_ste(k,:),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
    statelabels{k} = [sort{k},':DMN'];
    h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
end
% grid on;
title('PSD');
xlabel('Frequency','FontSize', 8);
for k=1:2,h(k).DisplayName=[sort{k},':DMN'];end
leg=legend(h,'Location','EastOutside');
 
%%coh
subplot(2,1,2)
for k=1:2
    if mod(k,2)==1
         ls{k} = '-';
     else
         ls{k}='--';
    end
    % shadedErrorBar(0.5*f_band,psd_all(k,:),psd_ste(k,:));hold on;
    shadedErrorBar(0.5*f_band,coh_all(k,:),coh_ste(k,:),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
    statelabels{k} = [sort{k},':DMN'];
    h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
end
% grid on;
title('coh');
xlabel('Frequency','FontSize', 8);
for k=1:2,h(k).DisplayName=[sort{k},':DMN'];end
leg=legend(h,'Location','EastOutside');
 
print('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/diff','-dpng')
 
%%%%%%%%%
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';
load([savebase '/mydata.mat'])
newcoh=squeeze(mean(squeeze(mean(coh(:,2,:,:,:),1)),1));
figure
heatmap(newcoh,'FontSize',5);
print('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/children_coh','-dpng')
 
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/store';
load([savebase '/mydata.mat'])
newcoh=squeeze(mean(squeeze(mean(coh(:,2,:,:,:),1)),1));

figure
heatmap(newcoh,'FontSize',5);

print('/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/adults_coh','-dpng')
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HMM Temporal statistics
savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/adults/store';
[hmm_input,T_input,hmm_input_files]=getdata(1,'adults',0);
ind=logical([1 0 0 1 1 1 1 1 0 1 1 1 0 1]);

savebase = '/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/SPM_ROI_DATA/train/children/store';
[hmm_input,T_input,hmm_input_files]=getdata(1,'children',0);
ind=logical([0 1 0 1 0 1 1 1 0 1 1 1 0]);

hmm_input_files=hmm_input_files(ind);
X=hmm_input(ind);
T=T_input(ind);

options = struct();
options.K = 12;
options.order =  0;
options.covtype = 'full';
options.zeromean = 1;
options.embeddedlags = -7:7; 
options.pca = 68 * 2;
options.standardise = 1;
options.Fs = 250;
options.verbose = 1;
options.initrep = 1;
options.initcyc = 1;
options.cyc = 30;

tic
[hmm_tde,Gamma_tde] = hmmmar(X,T,options);
toc/60
 

[viterbipath] = hmmdecode(X,T,hmm_tde,1);

% disttoplot = plotMDS_states(hmm_tde);
% [~,new_state_ordering] = sort(disttoplot(:,1));

% hmm_new = hmm_permutestates(hmm_tde,new_state_ordering);


% for i=1:hmm.K
%     hmm_new.gamma_tde(:,i)=hmm.gamma_tde(:,new_state_ordering(i));
% end
% hmm_new.statepath=hmm.statepath;
% % X=cell2mat(hmm_input');
% % Gamma_tde=hmm.gamma_tde;
% scan_T=cell2mat(T_input);
% Gamma = hmm_new.gamma_tde;
% sample_rate = 250;
% statepath=hmm_new.statepath;
% viterbipath=hmm.statepath;
scan_T=cell2mat(T);
options=[];
options.Fs=250;
% FO = getFractionalOccupancy( Gamma_tde, scan_T, options,2);
% IT = getStateIntervalTimes( Gamma_tde, scan_T, [],[],[],false);
% ITmerged = cellfun(@mean,IT);clear IT
% LT = getStateLifeTimes( Gamma_tde, scan_T, [],[],[],false);
% LTmerged = cellfun(@mean,LT); clear LT

FO = getFractionalOccupancy( viterbipath, scan_T, options,2);
IT = getStateIntervalTimes( viterbipath, scan_T, [],[],[],false);
ITmerged = cellfun(@mean,IT);clear IT
LT = getStateLifeTimes( viterbipath, scan_T, [],[],[],false);
LTmerged = cellfun(@mean,LT); clear LT
sample_rate = 250;
% Make summary figures概率分布图
fontsize = 10;
color_scheme = set1_cols();
figure
distributionPlot(FO,'showMM',2,'color',{color_scheme{1:size(FO,2)}});
set(gca,'YLim',[0 1.1*max(FO(:))],'FontSize',fontsize)
title('Fractional Occupancy');
xlabel('RSN-State');ylabel('Proportion');grid on;
 
print([savebase '/temporalstats_FO'],'-dpng')


figure
distributionPlot(LTmerged ./ sample_rate * 1000,'showMM',2,'color',{color_scheme{1:size(FO,2)}})
title('Life Times');
xlabel('RSN-State');ylabel('Time (ms)');grid on;
YL = 1.1*max(LTmerged(:))./ sample_rate * 1000;
set(gca,'YLim',[0 5000],'FontSize',fontsize,'FontSize',fontsize);
 
print([savebase '/temporalstats_LT'],'-dpng')


% figure
% distributionPlot(log10(ITmerged ./ sample_rate),'showMM',2,'color',{color_scheme{1:size(FO,2)}})
% title('Interval Times');
% xlabel('RSN-State');ylabel('Time (secs)');grid on
% YL(2) =10* max(mean(log10(ITmerged ./ sample_rate)));
% YL(1) = min(squash(log10(ITmerged ./ sample_rate)));
% set(gca,'YLim',YL,'FontSize',fontsize)
% set(gca,'YTick',log10([0.05,0.1,0.5,1,5,10]))
% y_labels = get(gca,'YTickLabel');
% for i=1:length(y_labels)
%     y_labels{i}=num2str(10.^(str2num(y_labels{i})),1);
% end
% set(gca,'YTickLabels',y_labels);
 
% print([savebase '/temporalstats_IT_logscale'],'-dpng')


figure;
distributionPlot(ITmerged ./ sample_rate,'showMM',2,'color',{color_scheme{1:size(FO,2)}})
title('Interval Times');
xlabel('RSN-State');ylabel('Time (secs)');grid on
YL(2) =30* max(mean((ITmerged ./ sample_rate)));
YL(1) = 0;
set(gca,'YLim',YL,'FontSize',fontsize)
 
print([savebase '/temporalstats_IT'],'-dpng')

close all;
 
