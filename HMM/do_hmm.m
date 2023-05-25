%%%%%%%%%%%%%%%%%%%%%%%%%%%%% do-hmm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hmm,hmm_output_file]=do_hmm(n,sort,h)
    [hmm_input,T_input,hmm_input_files]=getdata(n,sort,h);

%reduce source_leakage
if h
    for i=1:length(hmm_input)
        data=hmm_input{i};
        data = leakcorr(data,size(data,1),14);
        % data = ROInets.remove_source_leakage(hmm_input{i}','symmetric');
        hmm_input{i}=data;
    end
end

% figure
% imagesc(corr(a)+diag(nan(68,1)))
% axis square
% colorbar
% set(gca,'CLim',[-1 1])
% title('Raw correlation before leakage correction')
 
% figure
% imagesc(corr(data)+diag(nan(68,1)))
% axis square
% colorbar
% set(gca,'CLim',[-1 1])
% title('Raw correlation after leakage correction')
 
    [hmm_output_file,~,~]=fileparts(hmm_input_files{1});

    options = struct();
    options.K = 12;
    options.order =  0;
    options.covtype = 'full';
    options.zeromean = 1;
    options.embeddedlags = -7:7; 
    options.pca = 68 * 2; 
    options.standardise = 1;
    options.Fs = 250;
    % options.Fs = 600;
    % show progress?
    options.verbose = 1;
    options.initrep = 1; 
    options.initcyc = 1; 
    options.cyc = 30; 

    [hmm_tde,Gamma_tde,~,~,~,~,fehist] = hmmmar(hmm_input,T_input,options);
    [Gamma,~] = hmmdecode(hmm_input,T_input,hmm_tde,0) ; %前向-后向算法解码
    [viterbipath] = hmmdecode(hmm_input,T_input,hmm_tde,1); %Viterbi 算法来解码

    embed=[];
    embed.do  = 1;%
    embed.rectify  = false;
    embed.centre_freq = 14; 

    normalisation='voxelwise';
    logtrans=0;
    subj_inds=[];
    freq_ind=[];

    for subnum = 1:length(hmm_input_files)
        D = spm_eeg_load(hmm_input_files{subnum});            
        embed.tres=1/D.fsample;
        data = osl_teh_prepare_data(D,normalisation,logtrans,freq_ind,embed);
        
        % if max_ntpts>0
        %     ntpts=min(max_ntpts,size(data,2));
        %     data = data(:,1:ntpts);
        % end   
        subj_inds  = [subj_inds, subnum*ones(1,size(data,2))];
    end
    D = spm_eeg_load(hmm_input_files{1});

    hmm= struct();
    hmm.hmm=hmm_tde;
    hmm.gamma_tde=Gamma_tde;
    hmm.fehist=fehist;
    hmm.gamma=Gamma;
    hmm.statepath=viterbipath;
    hmm.options = options;
    hmm.data_files = hmm_input_files;
    hmm.K=options.K;
    hmm.fsample = D.fsample;
    hmm.subj_inds=subj_inds;

end