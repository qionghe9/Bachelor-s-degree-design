function [psd,coh,f] = loadMTspect(savebase,h,new_state_ordering,permuteSTCs)
if nargin<4
    permuteSTCs = false; % note this is a flag for testing whether the estimation is biased; if true, it scrambles the STCs
end
if permuteSTCs
    PMstring = '_permuted';
else
    PMstring = '';
end

hard=h; % 1 for hard state assignment, 0 for soft
if hard
    fulldir = [savebase '/state_netmats_mtsess_2_vn0_hard_global0.mat'];
else
    fulldir = [savebase '/state_netmats_mtsess_2_vn0_soft_global0.mat'];
end

load(fulldir);
state_netmats=state_netmats_mtsess;
NK=length(state_netmats{1}.state);
num_nodes=size(state_netmats{1}.state{1}.netmat,1);
num_freqs=length(state_netmats{1}.state{1}.spectramt.f);
nsubjects=length(state_netmats);

% hmmfile = [basedir,'/hmm_1to45hz/hmm',templatestring,'_parc_giles_symmetric__pcdim80_voxelwise_embed14_K',int2str(K),'_big1_dyn_modelhmm.mat'];
% load(hmmfile,'new_state_ordering');

psd=zeros(nsubjects,NK,num_freqs,num_nodes,num_nodes);
coh = zeros(nsubjects,NK,num_freqs,num_nodes,num_nodes);
for ss=1:length(state_netmats)
    for kk=1:length(state_netmats{1}.state)
        try           
        psd(ss,kk,:,:,:)=state_netmats{ss}.state{new_state_ordering(kk)}.spectramt.psd; 
        coh(ss,kk,:,:,:)=state_netmats{ss}.state{new_state_ordering(kk)}.spectramt.coh; 
        catch
        psd(ss,kk,:,:,:)=NaN; 
        coh(ss,kk,:,:,:)=NaN;
        end    
    end
    % add global on the end
    %psds(ss,kk+1,:,:,:)=state_netmats{ss}.global.spectramt.psd;
end

f = state_netmats{1}.state{1}.spectramt.f;

end
