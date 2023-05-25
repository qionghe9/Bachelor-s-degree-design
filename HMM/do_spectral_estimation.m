function []=do_spectral_estimation(hmm_output_file,n)

load([hmm_output_file,int2str(n),'th_hmm_result.mat'],'hmm')
storage_dir=[hmm_output_file 'store/'];
if ~exist(storage_dir,'file')
    mkdir(storage_dir);
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate spectra and cross spectra using multitaper on concatenated data

do_run=1;

S=[];
S.parcellated_filenames=hmm.data_files; %
S.normalisation='voxelwise';
S.assignment='soft';
% S.assignment='soft';
S.global_only=false;
S.embed.do=0;
S.embed.rectify=false;

if hmm.fsample==250
    freq_range=[1 45];
else
    freq_range=[1 160];
end

S.netmat_method=@netmat_spectramt;
S.netmat_method_options.fsample=hmm.fsample;%
S.netmat_method_options.fband=freq_range;
S.netmat_method_options.type='coh';
S.netmat_method_options.full_type='full';
S.netmat_method_options.var_normalise=false;
S.netmat_method_options.reg=2; % higher is less reg
S.netmat_method_options.order=0;

if do_run

    [ state_netmats_mt ] = hmm_state_netmats_teh_concat( hmm, S );

    save([storage_dir 'state_netmats_mt' num2str(floor(S.netmat_method_options.reg)) ...
        '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment '_' ...
        'global' num2str(S.global_only)], '-v7.3', 'state_netmats_mt');
% % % else
% % %     load([storage_dir '/state_netmats_mt_' num2str(floor(S.netmat_method_options.reg)) ...
% % %     '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment  '_' ...
% % %     'global' num2str(S.global_only)],'state_netmats_mt');

end

%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate spectra and cross spectra using multitaper on each session separately

do_run=1;

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

if do_run

    [ state_netmats_mtsess ] = hmm_state_netmats_teh( hmm, S );

    save([storage_dir 'state_netmats_mtsess_' num2str(floor(S.netmat_method_options.reg)) ...
        '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment '_' ...
        'global' num2str(S.global_only)], '-v7.3', 'state_netmats_mtsess');

% else

%     load([storage_dir '/state_netmats_mtsess_' num2str(floor(S.netmat_method_options.reg)) ...
%     '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment  '_' ...
%     'global' num2str(S.global_only)],'state_netmats_mtsess');

end

end