% function []=else_plot(resultfiles,hmm_input,T_input,Gamma_tde,hmm_tde)
function []=else_plot(resultfiles,hmm_input,T_input)

options = struct();
options.K = 12;
options.order =  0;
options.covtype = 'full';
options.zeromean = 1;
options.embeddedlags = -7:7; % 15 lags are used from -7 to 7, this defines the length of the modelled autocorrelations
options.pca = 68 * 2; % twice the number of regions (see the logic of this on Vidaurre et al. 2018)
% standardize each region
options.standardise = 1;
% Sampling frequency in the data
options.Fs = 250;
% show progress?
options.verbose = 1;

% some options relative to training, we will make it cheap to run:
options.initrep = 1; % to make it quicker - leave by default otherwise
options.initcyc = 1; % to make it quicker - leave by default otherwise
options.cyc = 30; % to make it quicker - leave by default otherwise


[hmm_tde,Gamma_tde] = hmmmar(hmm_input,T_input,options);

options = struct();
options.fpass = [1 45]; 
options.tapers = [4 7]; % internal multitaper parameter
options.Fs = 250; 
options.win = 10 * options.Fs; % window length, related to the level of detail of the estimation;
options.embeddedlags = -7:7;

spectra_tde2 = hmmspectramt(hmm_input,T_input,Gamma_tde,options);

save(resultfiles,'spectra_tde2','-v7.3')

% channels_prim_visual_cortex = [26 27];%ROI ID
% plot_hmmspectra (spectra_tde,[],[],[],[],channels_prim_visual_cortex);
%plot_hmmspectra (spectra_tde,[],[],[],[]);

% %%%%%%
% params_fac = struct();
% params_fac.Base = 'coh';
% params_fac.Method = 'NNMF';
% params_fac.Ncomp = 2; % set to a higher value (4) to pull out more detailed frequency modes

% tic
% [spectral_factors,spectral_profiles] = spectdecompose(spectra_tde,params_fac);
% toc
% %save(results_file,'spectral_factors','spectral_profiles','-append')

% atlasfile='/GPFS/liuyunzhe_lab_permanent/heqiong/fsaverage/label/lh.aparc.gii';
% g=gifti(atlasfile);

% p = parcellation(atlasfile); % load the parcellation

% net_mean = zeros(68,hmm_tde.train.K);
% for k = 1:length(spectra_tde.state)
%     net_mean(:,k) =  diag(squeeze(abs(spectral_factors.state(k).psd(1,:,:))));
% end
% net_mean = zscore(net_mean); % show activations for each state relative to the state average
% p.osleyes(net_mean); % call osleyes


%%%%%%%
t = 3001:5001; % some arbitrary time segment
figure
area(t/250,Gamma_tde(t,:),'LineWidth',2);  xlim([t(1)/250 t(end)/250])
xlabel('Time'); ylabel('State probability')
title('TDE-HMM' )
saveas(gcf,'arbitrary_timesegment_states.png')


figure
imagesc(getTransProbs(hmm_tde)); colorbar
xlabel('From state'); ylabel('To state'); axis image
title('TDE-HMM' )
saveas(gcf,'TransProbs.png')


T=cell2mat(T_input)';
lifetimes_tde = getStateLifeTimes (Gamma_tde,T,hmm_tde.train,[],[],false);
lifetimes_tde = lifetimes_tde(:)'; lifetimes_tde = cell2mat(lifetimes_tde);
figure
hist(lifetimes_tde/250,100); xlim([0 0.5]); ylim([0 10000])
xlabel('Life times'); ylabel('No. of visits')
title('TDE-HMM')
saveas(gcf,'StateLifeTimes.png')
    %The TDE-HMM, which has a tendency to focus on slower frequencies, has the longest life times.

    % Intervals = getStateIntervalTimes (Gamma_tde,hmm_tde.train); 
    % Intervals = Intervals(:)'; Intervals = cell2mat(Intervals);
    % figure
    % hist(Intervals/250,100); xlim([0 0.5]); ylim([0 10000])
    % xlabel('Intervals'); ylabel('No. of visits')
    % title('TDE-HMM')
    % saveas(gcf,'Intervals.png')

    

end