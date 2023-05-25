function [hmm, hmmfname, hmmoptions, settings_prepare] = run_full_hmm_pipeline_neuron2020edit(S)

% [hmm, hmmfname, hmmoptions, settings_prepare] = run_full_hmm_pipeline(S)

try preproc_name=S.preproc_name; catch, error('S.preproc_name needed'); end
try session_name=S.session_name; catch, error('S.session_name needed'); end
try sort=S.sort;catch,error('S.sort needed');end
try spmfilesdir=S.spmfilesdir; catch, error('S.spmfilesdir needed'); end
try prep_sessions_to_do=S.prep_sessions_to_do; catch, error('S.prep_sessions_to_do needed'); end
try hmm_sessions_to_do=S.hmm_sessions_to_do; catch, hmm_sessions_to_do=prep_sessions_to_do; end
try do_hmm=S.do_hmm; catch, do_hmm=1; end
try do_spectral_estimation=S.do_spectral_estimation; catch, do_spectral_estimation=1; end
try num_embeddings=S.num_embeddings; catch, num_embeddings  = 12; end
try hmm_name=S.hmm_name; catch, hmm_name  = ''; end
try freq_range=S.freq_range; catch, freq_range=[1 45]; end

try parcellations_dir=S.parcellations_dir; catch, parcellations_dir='/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations'; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
settings_prepare=[];
settings_prepare.parcellation.orthogonalisation='symmetric';
settings_prepare.dirname=spmfilesdir;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setup HMM
%% get hmm input files
%hmm_mode='envelope';
hmm_mode='raw';

% build file names for signflip_parcellated_files for HMM input:

switch settings_prepare.parcellation.orthogonalisation
    case 'innovations_mar'
        parc_prefix   = [settings_prepare.parcellation.orthogonalisation num2str(settings_prepare.parcellation.innovations_mar_order) '_'];
    otherwise
        parc_prefix   = [settings_prepare.parcellation.orthogonalisation '_'];
end

signflipped_files={};
unsignflipped_files={};
%enveloped_files={};
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
datadir=fullfile(basedir,'meg-data','ROI_DATA',session_name);
fileList=dir(fullfile(datadir,sort,'*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
fif_name={};
for s=1:length(fileNames)
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    unsignflipped_files{s}=fullfile(spmfilesdir,strcat(numstr,'.mat'));
    signflipped_files{s}=fullfile(spmfilesdir,strcat('sfold_',numstr,'.mat'));
    %enveloped_files{ss}=fullfile(spmfilesdir,strcat('h_',numstr,'.mat'));
end

% check signflipped_files exist
%signflipped_files_ext = cellfun(@strcat,signflipped_files,repmat({'.mat'},1,length(signflipped_files)),'uniformoutput',0);
signflipped_files=unique(signflipped_files);
isfile = cell2mat(cellfun(@exist,signflipped_files,repmat({'file'},1,length(signflipped_files)),'uniformoutput',0))>0;

disp(['Number of sessions is ' num2str(length(signflipped_files))]);
if (any(~isfile))
    warning('Invalid signflipped_files');
end

signflipped_files=signflipped_files(isfile);
disp(['Number of sessions is ' num2str(length(signflipped_files))]);


D = spm_eeg_load(unsignflipped_files{1});
switch hmm_mode   
    case 'raw'    
        if D.fsample==600
            hmm_input_spm_files=unsignflipped_files;
        else
            hmm_input_spm_files=signflipped_files;
        end

    case 'envelope'    
        hmm_input_spm_files=enveloped_files;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HMM

hmmoptions=[];

switch hmm_mode
    
    case 'raw'
        % use tideh (time delay embedded HMM) on raw data        

        % dimensionality reduction
        % D=spm_eeg_load(hmm_input_spm_files{1});
        % nparcels=size(D.parcellation.weights,2); %注意unsignflipped_files的行列不是嵌入型的
        hmmoptions.prepare.pcadim           = 136; %注意是否区分左右脑区

        % embedding 
        hmmoptions.prepare.embed.do         = 1;
        hmmoptions.prepare.embed.num_embeddings = num_embeddings;
           
    case 'envelope'        
        
        % no dimensionality reduction
        hmmoptions.prepare.pcadim           = 0;
       
        % no embedding this time
        hmmoptions.prepare.embed.do         = 0;
        hmmoptions.prepare.embed.num_embeddings = 0;
end

% K defines the number of states
if isfield(S,'K')
    hmmoptions.hmm.K                    = S.K;
else
    hmmoptions.hmm.K                    = 12;
end

if isfield(S,'templateHMM') 
    temp = load(S.templateHMM);
    hmmoptions.hmm.hmm              = temp.hmm;
    hmmoptions.hmm.updateObs = 0;
    hmmoptions.hmm.BIGcyc = 1; % this the critical param: it means there is just one call to hmmdecode
end

hmmoptions.prepare.normalisation    = 'voxelwise';
hmmoptions.prepare.whiten           = 1; 
hmmoptions.prepare.savePCmaps       = 0;
%hmmoptions.prepare.max_ntpts        = 40000;

hmmoptions.hmm.dynamic_model_type   = 'hmm';
if isfield(S,'dynamic_model_type')
    hmmoptions.hmm.dynamic_model_type   = S.dynamic_model_type;
end
%hmmoptions.hmm.dynamic_model_type   = 'vbrt';

hmmoptions.hmm.initcyc              = 60;
hmmoptions.hmm.initrep              = 4;

hmmoptions.hmm.big                  = 1;
hmmoptions.hmm.BIGNbatch            = 10;
hmmoptions.hmm.name                 = hmm_name;

hmmoptions.output.method       = 'pcorr';
hmmoptions.output.use_parcel_weights = 0;
hmmoptions.output.assignment   = 'hard';

% setup filenames

hmmoptions.prepare.filename     = ['hmm' hmmoptions.hmm.name ...
                                '_parc_' parc_prefix ...                                
                                '_pcdim' num2str(hmmoptions.prepare.pcadim) ...
                                '_' num2str(hmmoptions.prepare.normalisation) ...
                                '_embed' num2str(hmmoptions.prepare.embed.num_embeddings)];
hmmoptions.hmm.filename        = [hmmoptions.prepare.filename ...
                                '_K' num2str(hmmoptions.hmm.K)...
                                '_big' num2str(hmmoptions.hmm.big)...
                                '_dyn_model' hmmoptions.hmm.dynamic_model_type];
hmmoptions.output.filename     = [hmmoptions.hmm.filename '_output'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% run HMM
hmm = [];hmmfname = [];
if do_hmm==1

    hmmoptions.todo.prepare  = 1;
    hmmoptions.todo.hmm      = 1;
    hmmoptions.todo.output   = 0;                                                                       
    
    hmmoptions.hmmdir = [settings_prepare.dirname 'hmm_' num2str(freq_range(1)) 'to' num2str(freq_range(2)) 'hz/'];
    [HMMresults_raw_flips] = teh_groupinference_parcels(hmm_input_spm_files,hmmoptions);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% load previously run HMM

    hmmdir=[settings_prepare.dirname 'hmm_' num2str(freq_range(1)) 'to' num2str(freq_range(2)) 'hz/'];
    load([hmmdir hmmoptions.hmm.filename]);

    hmmfname=[hmmdir hmmoptions.hmm.filename];

    % add some settings to hmm for use later

    % D=spm_eeg_load(hmm.data_files{1});
    % hmm.parcellation= D.parcellation;
    % hmm.parcellation.file = strrep(hmm.parcellation.S.parcellation,'/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs',...
    %     osldir);%替换函数 strrep(str,old,new)

    % sres=nii.get_spatial_res(hmm.parcellation.file);
    % gridstep=sres(1);

    % hmm.parcellation.mask=[osldir '/std_masks/MNI152_T1_' num2str(gridstep) 'mm_brain'];

    save([hmmdir hmmoptions.hmm.filename],'hmm');
end
%%
%总数据或每个被试的 *每个状态* 进行do_spectral_estimation
if do_spectral_estimation
    
    hmmdir=[settings_prepare.dirname 'hmm_' num2str(freq_range(1)) 'to' num2str(freq_range(2)) 'hz/'];
    load([hmmdir hmmoptions.hmm.filename]);

    hmmfname=[hmmdir hmmoptions.hmm.filename];
    
    storage_dir=[hmmfname '_store/'];

    mkdir(storage_dir);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Estimate spectra and cross spectra using multitaper on concatenated data

    do_run=1;

    S=[];
    S.parcellated_filenames=hmm.data_files; %
    S.normalisation='voxelwise';
    S.assignment='hard';
    S.global_only=false;
    S.embed.do=0;
    S.embed.rectify=false;

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

        save([storage_dir '/state_netmats_mt' num2str(floor(S.netmat_method_options.reg)) ...
            '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment '_' ...
            'global' num2str(S.global_only)], '-v7.3', 'state_netmats_mt');
    else
        load([storage_dir '/state_netmats_mt_' num2str(floor(S.netmat_method_options.reg)) ...
        '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment  '_' ...
        'global' num2str(S.global_only)],'state_netmats_mt');

    end

    %%%%%%%%%%%%%%%%%%%%%%%%
    %% Estimate spectra and cross spectra using multitaper on each session separately

    do_run=1;

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

    if do_run

        [ state_netmats_mtsess ] = hmm_state_netmats_teh( hmm, S );

        save([storage_dir '/state_netmats_mtsess_' num2str(floor(S.netmat_method_options.reg)) ...
            '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment '_' ...
            'global' num2str(S.global_only)], '-v7.3', 'state_netmats_mtsess');

    else

        load([storage_dir '/state_netmats_mtsess_' num2str(floor(S.netmat_method_options.reg)) ...
        '_vn' num2str(S.netmat_method_options.var_normalise) '_' S.assignment  '_' ...
        'global' num2str(S.global_only)],'state_netmats_mtsess');

    end
end



