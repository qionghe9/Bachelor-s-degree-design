function [ spm_files_preproc,template_subj ] = prep_parcellated_data( settings )

try freq_range=settings.freq_range; catch, error('freq_range'); end
% try parcellation_to_use=settings.parcellation.parcellation_to_use; catch, error('parcellation_to_use'); end

try sessions_to_do=settings.sessions_to_do; catch, error('sessions_to_do'); end

try num_iters=settings.signflip.num_iters; catch, num_iters=500; end
try num_embeddings=settings.signflip.num_embeddings; catch, num_embeddings=10; end
try do_signflip_diagnostics=settings.do_signflip_diagnostics; catch, do_signflip_diagnostics=settings.do_signflip; end

try parcellated_files=settings.parcellated_files; catch, error('parcellated_files'); end

try settings.parcellation.orthogonalisation; catch, settings.parcellation.orthogonalisation='innovations_mar'; end
try innovations_mar_order=settings.parcellation.innovations_mar_order; catch, innovations_mar_order=14; end
try sort1=settings.sort1; catch, error('sort1'); end
try sort2=settings.sort2; catch, error('sort2'); end

spm_files_preproc=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% enveloping

if settings.do_hilbert % needed for HMM on envelopes
    hilbert_files={};
    for ss=1:length(sessions_to_do)
        S=[];
        S.D = parcellated_files{ss};
        S.winsize = 1/40; %secs
        %S.winsize = 0; %secs
        S.downsample=0;
        S.remove_edge_effects=1;
        S.prefix  ='h';
        hilbert_files{ss} = osl_hilbenv(S);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sign flip stuff

if settings.do_signflip || do_signflip_diagnostics

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % find template subject
    clear state_netmats_cov_preflipped;
   
    %%%%%%%%%%
    % establish a good template subject
    S=[];
    S.concat = [];
    S.concat.protocol='none';
    S.concat.embed.do=1;
    S.concat.embed.num_embeddings=num_embeddings;
    S.concat.embed.rectify=false;
    S.concat.whiten=1;
    S.concat.normalisation='voxelwise';
    S.concat.pcadim=-1;
    S.netmat_method=@netmat_cov;

    state_netmats_cov_preflipped = hmm_full_global_cov( parcellated_files, S );
else
    state_netmats_cov_preflipped = [];
end

clear template_subj;

% assess which subject is the best template:
state_netmats=state_netmats_cov_preflipped;  

modes={'none','abs'};
diag_offset=15;
if ~isfield(settings,'templatesubj') %判断结构体是否存在变量
    metric_global=zeros(length(state_netmats),length(state_netmats),length(modes));
           
    for mm=1:length(modes)
        for subj=1:length(state_netmats)
           for subj2=1:length(state_netmats)
                if subj2~=subj
                    metric_global(subj,subj2,mm)=matrix_distance_metric(state_netmats{subj}.global.netmat_full, state_netmats{subj2}.global.netmat_full,diag_offset,modes{mm},[]);
                end
           end
        end
    end

    tmp=sum(metric_global(:,:,2),2);

    [~, template_subj]=max(tmp);%返回每一列最大值的索引
else
    template_subj = length(parcellated_files);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  do actual sign flip
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
bfdir=fullfile(basedir,'meg-data','SPM_ROI_DATA',sort1,sort2);
if settings.do_signflip
    disp("do_signflip")
    if 1
        % sign flipping settings
        S=[];
        S.roinets_protocol=settings.parcellation.orthogonalisation;
        S.innovations_mar_order = innovations_mar_order;            
        S.Ds=parcellated_files;
        S.num_iters=num_iters;
        S.prefix='sfold_';
        S.num_embeddings=num_embeddings;
        S.subj_template=template_subj;
        [ signflipped_files_out, sign_flip_results ] = find_sign_flips( S );

        sign_flip_results.signflipped_files=signflipped_files_out;
        sign_flip_results.energies_group=mean(sign_flip_results.energies,2);
        sign_flip_results.energies=sign_flip_results.energies(1:20:end,:);
        save(fullfile(bfdir,'sign_flip_results'),'-struct','sign_flip_results','-v7.3');

    else

        S=[];
        S.Ds = parcellated_files;
        S.prefix = 'sf_';
        S.options=[];
        S.options.maxlag=4; % max lag to consider, considering that we are including lagged autocovariance matrices in the calculation (default to 4).
        S.options.noruns=50; % how many random initialisations will be carried out (default to 50).
        S.options.maxcyc=200; % for each initialization, maximum number of cycles of the greedy algorithm (default to 100 * N * no. of channels).
        [Dsnew, flips, scorepath] = osl_resolve_sign_ambiguity(S);

    end

end
disp("go on")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot some sign flip diagnostics

if do_signflip_diagnostics
    disp("do_signflip_diagnostics")
    sign_flip_results=load(fullfile(bfdir,'sign_flip_results'));

    signflip_parcellated_files={};
    for ss = 1:length(sessions_to_do)
        [~,name,~]=fileparts(parcellated_files{ss});
        signflip_parcellated_files{ss}=fullfile(bfdir,strcat('sfold_',name,'.mat'));
    end
    
    S=[];
    S.concat = [];
    S.concat.protocol=settings.parcellation.orthogonalisation;
    S.innovations_mar_order = innovations_mar_order;            
    S.concat.embed.do=1;
    S.concat.embed.num_embeddings=num_embeddings;
    S.concat.embed.rectify=false;
    S.concat.whiten=1;
    S.concat.normalisation='voxelwise';
    S.concat.pcadim=-1;
    S.netmat_method=@netmat_cov;

    [ state_netmats_cov_signflipped ] = hmm_full_global_cov( signflip_parcellated_files, S );

    subj_template_no=template_subj;

    print_fname=[bfdir '/sign_flip_plot'];

    plot_sign_flip_results(state_netmats_cov_preflipped,state_netmats_cov_signflipped, subj_template_no, freq_range, sign_flip_results, print_fname);

    disp('Sign flip diagonostic plot saved to:');
    disp(print_fname);

    spm_files_preproc=signflip_parcellated_files;
end

disp("end")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

