%%%%%%%%%%%%%%%%%%%%%%%%%after filter%%%%%%%%%%%%%%%%%%%%%%

pwd 
cd /GPFS/liuyunzhe_lab_permanent/heqiong/osl/osl-core/ %更改
osl_startup;

%%%%%%%%%%convert fif to spm
basedir='/GPFS/liuyunzhe_lab_permanent/heqiong/';
datadir=fullfile(basedir,'meg-data','raw_filter_data','train');
sort={'adults','children'};

for i=1:length(sort)
    fileList=dir(fullfile(datadir,sort{i},'*.fif'));
    fileNames = {fileList.name};
    filefolder={fileList.folder};
    spm_roi_data={};
    fif_name={};
    for s=1:length(fileNames)
        fif_name{s}=fullfile(filefolder{s},fileNames{s});
        % [~, name] = fileparts(fif_name{s});
        numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
        spm_path=fullfile(basedir,'meg-data','spm_raw_filter_data','train',sort{i},numstr);
        % if ~exist(spm_path,'dir')
        %     mkdir(spm_path)
        % end
        spm_roi_data{s}=[spm_path '.mat'];
        S2=[];
        S2.outfile = spm_roi_data{s};
        S2.trigger_channel_mask = '0000000000111111';
        osl_import(fif_name{s},S2);
    end
end

% A='/GPFS/liuyunzhe_lab_permanent/heqiong/meg-data/MEG_resting/adults/sub_067/NeuroData/MEG/rest1_tsss.fif'
% spm_path=fullfile('/GPFS/liuyunzhe_lab_permanent/heqiong/a');

% S2=[];
% S2.outfile = spm_path;
% S2.trigger_channel_mask = '0000000000111111';
% osl_import(A,S2);

% D=spm_eeg_load(spm_path)
% has_montage(D)

%%%%%%%%%spm load
sort1='train';
sort='adults';
fileList=dir(fullfile(datadir,sort,'*.fif'));
fileNames = {fileList.name};
filefolder={fileList.folder};
spm_roi_data={};
fif_name={};
for s=1:2
    fif_name{s}=fullfile(filefolder{s},fileNames{s});
    numstr=strjoin(regexp(fif_name{s}, '\d+', 'match'),'_');
    spm_path=fullfile(basedir,'meg-data','spm_raw_filter_data','train',sort,numstr);
    spm_roi_data{s}=[spm_path '.mat'];
end

%%%
D = spm_eeg_load(spm_roi_data{1});
has_montage(D) %0
spatial_basis_file ='/GPFS/liuyunzhe_lab_permanent/heqiong/osl/parcellations/MNI152_T1_8mm_brain.nii';
spatial_basis = nii.load(spatial_basis_file);
size(spatial_basis)
osleyes(spatial_basis_file)
p = parcellation(spatial_basis_file);
p.plot
size(p.binarize) % Binarize the voxel assignments
size(p.to_matrix(p.binarize)) % Reshape from volume to matrix representation
D = ROInets.get_node_tcs(D,p.to_matrix(p.binarize),'pca'); % Use ROInets to get parcel timecourses

has_montage(D)

D = D.montage('switch',3);
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
%%%%%%%%%%%%%%%%
S = [];
S.D = spm_roi_data{1};
S.mri = spatial_basis_file;
S.useheadshape = 1;
S.use_rhino = 0;
S.forward_meg = 'Single Shell';
S.fid.label.nasion = 'Nasion';
S.fid.label.lpa = 'LPA';
S.fid.label.rpa = 'RPA';
D=osl_headmodel(S);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Beamform:
S                   = [];
S.modalities        = modalities;
S.timespan          = [0 Inf];
S.pca_order         = 120;
S.type              = 'Scalar';
S.inverse_method    = 'beamform';
S.prefix            = '';
S.modalities        = {'MEGPLANAR' 'MEGMAG'};
S.fuse              = 'meg';
S.pca_order         =64;
mni_coords          = osl_mnimask2mnicoords(spatial_basis_file);
D=osl_inverse_model(D,mni_coords,S);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parcellate 


   % setup bf file list
    bfdir = [dirname 'bfnew_' num2str(freq_range(1)) 'to' num2str(freq_range(2)) 'hz/'];

    bf_files=[];
    for ss = 1:length(sessions_to_do)        
       session = sessions_to_do(ss);
       [~, fname]=fileparts(spm_files{session});
       bf_files{session}=[bfdir fname];
       bf_files{session}=prefix(bf_files{session},'f');
    end

    %%%%%%%%%%%%
    % parcellate the data

    parcellated_Ds=[];

    for ss = 1:length(sessions_to_do)
        session = sessions_to_do(ss);
        S                   = [];
        S.D                 = bf_files{session};
        S.parcellation      = parc_file;
        S.orthogonalisation = settings.parcellation.orthogonalisation;
        S.innovations_mar_order = innovations_mar_order;
        S.method            = 'spatialBasis';
        S.normalise_voxeldata = 0;
        S.prefix=parc_prefix;
        [parcellated_Ds{ss},parcelWeights,parcelAssignments] = osl_apply_parcellation(S);

        parcellated_Ds{ss}.parcellation.weights=parcelWeights;
        parcellated_Ds{ss}.parcellation.assignments=parcelAssignments;

        parcellated_Ds{ss}.save;
    end

    % D=spm_eeg_load(options.Ds{1}); tmp=std(D(:,:,1),[],2);fslview(nii_quicksave(tmp,'davve'))
    % tmp=std(newDs{1}(:,:,1),[],2);fslview(ROInets.nii_parcel_quicksave(tmp,parcelAssignments,'dave'));

