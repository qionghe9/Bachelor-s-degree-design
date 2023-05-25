function [Gamma,Gammasum,Xi,LL,B] = hsinference(data,T,hmm,residuals,options,XX)
%
% inference engine for HMMs.
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure
% residuals in case we train on residuals, the value of those.
% XX        optionally, XX, as computed by setxx.m, can be supplied
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data
% LL        Log-likelihood
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T);
K = length(hmm.state);

mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;

if ~isfield(hmm,'train')
    if nargin<5 || isempty(options)
        error('You must specify the field options if hmm.train is missing');
    end
    hmm.train = checkoptions(options,data.X,T,0);
end
order = hmm.train.maxorder;

if iscell(data)
    data = cell2mat(data);
end
if ~isstruct(data)
    data = struct('X',data);
    data.C = NaN(size(data.X,1)-order*length(T),K);
end

if nargin<4 || isempty(residuals)
    ndim = size(data.X,2);
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

if ~isfield(hmm,'P')
    hmm = hmmhsinit(hmm);
end

if nargin<6 || isempty(XX)
    setxx;
end

Gamma = cell(N,1);
LL = zeros(N,1);
Gammasum = zeros(N,K);
if ~mixture_model
    Xi = cell(N,1);
else
    Xi = [];
end
B = cell(N,1);

n_argout = nargout;

ndim = size(residuals,2);
S = hmm.train.S==1;
regressed = sum(S,1)>0;

% Cache shared results for use in obslike
for k = 1:K
    setstateoptions;
    %hmm.cache = struct();
    hmm.cache.train{k} = train;
    hmm.cache.order{k} = order;
    hmm.cache.orders{k} = orders;
    hmm.cache.Sind{k} = Sind;
    hmm.cache.S{k} = S;
   
    if k == 1 && strcmp(train.covtype,'uniquediag')
        ldetWishB=0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    elseif k == 1 && strcmp(train.covtype,'uniquefull')
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
    elseif strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum = 0;
        for n=1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.state(k).Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape);
        end
        C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
    elseif strcmp(train.covtype,'full')
        ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
    end
    if ~isfield(train,'distribution') || ~strcmp(train.distribution,'logistic')
        hmm.cache.ldetWishB{k} = ldetWishB;
        hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
        hmm.cache.C{k} = C;
        hmm.cache.do_normwishtrace(k) = ~isempty(hmm.state(k).W.Mu_W);
    end
end

if hmm.train.useParallel==1 && N>1
    
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine
    parfor j = 1:N
        xit = [];
        Bt = [];  
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1);
        if order>0
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(j)-order,:)];
            if isfield(data,'C')
                C = [zeros(order,K); data.C(s0+1:s0+T(j)-order,:)];
            else
                C = NaN(size(R,1),K);
            end
        else
            R = residuals(s0+1:s0+T(j)-order,:);
            if isfield(data,'C')
                C = data.C(s0+1:s0+T(j)-order,:);
            else
                C = NaN(size(R,1),K);
            end
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else no_c = find(isnan(C(t:T(j),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints=slicer + s0 - order;
            XXt = XX(slicepoints,:); 
            if isnan(C(t,1))
                [gammat,xit,Bt] = nodecluster(XXt,K,hmm,R(slicer,:),slicepoints);
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model, xit = zeros(length(slicer)-1, K^2); end
                for i = 2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        xitr = gammat(i-1,:)' * gammat(i,:) ;
                        xit(i-1,:) = xitr(:)';
                    end
                end
                if n_argout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t>order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model, xi = [xi; xit]; end
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if n_argout>=4 
                ll = ll + sum(log(sum(Bt(order+1:end,:) .* gammat, 2))); 
            end
            if n_argout>=5, B{j} = [B{j}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else, t = no_c(1)+t-1;
            end
        end
        Gamma{j} = gamma;
        Gammasum(j,:) = gammasum;
        if n_argout>=4, LL(j) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        if ~mixture_model, Xi{j} = reshape(xi,T(j)-order-1,K,K); end
    end
    
else
    
    for j = 1:N % this is exactly the same than the code above but changing parfor by for
        Bt = [];  
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1);
        if order>0
            R = [zeros(order,size(residuals,2)); residuals(s0+1:s0+T(j)-order,:)];
            if isfield(data,'C')
                C = [zeros(order,K); data.C(s0+1:s0+T(j)-order,:)];
            else
                C = NaN(size(R,1),K);
            end
        else
            R = residuals(s0+1:s0+T(j)-order,:);
            if isfield(data,'C')
                C = data.C(s0+1:s0+T(j)-order,:);
            else
                C = NaN(size(R,1),K);
            end
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else no_c = find(isnan(C(t:T(j),1)));
            end
            if t > order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints=slicer + s0 - order;
            XXt = XX(slicepoints,:);
            if isnan(C(t,1))
                [gammat,xit,Bt] = nodecluster(XXt,K,hmm,R(slicer,:),slicepoints);
                if any(isnan(gammat(:)))
                    error('State time course inference returned NaN - Out of precision?')
                end
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model, xit = zeros(length(slicer)-1, K^2); end
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        xitr = gammat(i-1,:)' * gammat(i,:) ;
                        xit(i-1,:) = xitr(:)';
                    end
                end
                if nargout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t > order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model, xi = [xi; xit]; end
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if nargout>=4 
                ll = ll + sum(log(sum(Bt(order+1:end,:) .* gammat, 2)));
            end
            if nargout>=5, B{j} = [B{j}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else t = no_c(1)+t-1;
            end
        end
        Gamma{j} = gamma;
        Gammasum(j,:) = gammasum;
        if nargout>=4, LL(j) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        if ~mixture_model, Xi{j} = reshape(xi,T(j)-order-1,K,K); end
    end
end

% join
Gamma = cell2mat(Gamma);
if ~mixture_model, Xi = cell2mat(Xi); end
if n_argout>=5, B  = cell2mat(B); end

% orthogonalise = 1; 
% Gamma0 = Gamma; 
% if orthogonalise && hmm.train.tuda
%     T = length(Gamma) / length(T) * ones(length(T),1); 
%     Gamma = reshape(Gamma,[T(1) length(T) size(Gamma,2) ]);
%     Y = reshape(residuals(:,end),[T(1) length(T)]);
%     for t = 1:T(1)
%        y = Y(t,:)'; 
%        for k = 1:size(Gamma,3)
%            x = Gamma(t,:,k)';
%            b = y \ x;
%            Gamma(t,:,k) = Gamma(t,:,k) - (y * b)';
%        end
%     end
%     Gamma = reshape(Gamma,[T(1)*length(T) size(Gamma,3) ]);
%     Gamma = rdiv(Gamma,sum(Gamma,2));
%     Gamma = Gamma - min(Gamma(:));
%     Gamma = rdiv(Gamma,sum(Gamma,2));
% end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Xi,L] = nodecluster(XX,K,hmm,residuals,slicepoints)
% inference using normal foward backward propagation


if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic'); order=0;
else order = hmm.train.maxorder; end
T = size(residuals,1) + order;
Xi = [];

if nargin<5
    slicepoints=[];
end

% if isfield(hmm.train,'grouping') && length(unique(hmm.train.grouping))>1
%     i = hmm.train.grouping(n); 
%     P = hmm.P(:,:,i); Pi = hmm.Pi(:,i)'; 
% else 
%     P = hmm.P; Pi = hmm.Pi;
% end
P = hmm.P; Pi = hmm.Pi;

try
    if ~isfield(hmm.train,'distribution') || ~strcmp(hmm.train.distribution,'logistic')
        L = obslike([],hmm,residuals,XX,hmm.cache);
    else
        L = obslikelogistic([],hmm,residuals,XX,slicepoints);
    end
catch
    error('obslike function is giving trouble - Out of precision?')
end

if ~isfield(hmm.train,'id_mixture') && hmm.train.id_mixture
    Gamma = id_Gamma_inference(L,Pi,order);
    return
end


L(L<realmin) = realmin;

if hmm.train.useMEX 
    [Gamma, Xi, scale] = hidden_state_inference_mx(L, Pi, P, order);
    if any(isnan(Gamma(:))) || any(isnan(Xi(:)))
        clear Gamma Xi scale
        warning('hidden_state_inference_mx file produce NaNs - will use Matlab''s code')
    else
        return
    end
end

scale = zeros(T,1);
alpha = zeros(T,K);
beta = zeros(T,K);

alpha(1+order,:) = Pi.*L(1+order,:);
scale(1+order) = sum(alpha(1+order,:));
alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
for i = 2+order:T
    alpha(i,:) = (alpha(i-1,:)*P).*L(i,:);
    scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:) = alpha(i,:)/scale(i);
end

scale(scale<realmin) = realmin;

beta(T,:) = ones(1,K)/scale(T);
for i = T-1:-1:1+order
    beta(i,:) = (beta(i+1,:).*L(i+1,:))*(P')/scale(i);
    beta(i,beta(i,:)>realmax) = realmax;
end
Gamma = (alpha.*beta);
Gamma = Gamma(1+order:T,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

Xi = zeros(T-1-order,K*K);
for i = 1+order:T-1
    t = P.*( alpha(i,:)' * (beta(i+1,:).*L(i+1,:)));
    Xi(i-order,:) = t(:)'/sum(t(:));
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Gamma = id_Gamma_inference(L,Pi,order)
% inference for independent samples (ignoring time structure)

Gamma = zeros(T,K);
Gamma(1+order,:) = repmat(Pi,size(L,1),1) .* L(1+order,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

end

