function [nnmf_res,ss] = run_nnmf( S, niterations, summary_plot )
% function [nnmf_res,ss] = run_nnmf( S, niters, summary_plot )
%
%nargin 为当前函数输入变量的个数

if nargin < 3 || isempty( summary_plot )
    summary_plot = false;
end

if nargin < 2 || isempty( niterations )
    niterations = 10;
end

if summary_plot == true && niterations > 10
    warning('Summary plot is likely to be crowded with more than 10 iterations!');
end

disp(['summary_plot:' int2str(summary_plot)]);
S.do_plots = 0;

% Preallocate for SumSquare of residuls(拟合残差的平方)
ncomps = S.maxP;%4
nsamples = size( S.psds,3 );%90
ss = zeros( niterations, ncomps);

% Specify fit function, a unimodal gaussian
gauss_func = @(x,f) f.a1.*exp(-((x-f.b1)/f.c1).^2);%定义一个高斯分布函数

% Default fit options
options = fitoptions('gauss1'); %返回一个结构体，表示拟合一维高斯函数

% constrain lower and upper bounds
options.Lower = [0,1,0];
options.Upper = [Inf,nsamples,nsamples];

% Main loop
winning_value = Inf;
if summary_plot == true
    specs = zeros( ncomps, nsamples, niterations);
end

nnmf={};
i=1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );
i=i+1;
disp(["do nnmf:" int2str(i)]);
nnmf{i}.next_nnmf = my_teh_spectral_nnmf( S );


for ii = 1:i
    for jj = 1:ncomps
        f = fit( linspace(1,nsamples,nsamples)',nnmf{ii}.next_nnmf.nnmf_coh_specs(jj,:)', 'gauss1',options);
        resid = nnmf{ii}.next_nnmf.nnmf_coh_specs(jj,:) - gauss_func(1:nsamples,f);
        ss(ii,jj) = sum( resid.^2 );
    end

    if sum(ss(ii,:)) < winning_value
        nnmf_res = nnmf{ii}.next_nnmf;
        winning_value = sum(ss(ii,:));
    end

    if summary_plot == true
        specs(:,:,ii) = nnmf{ii}.next_nnmf.nnmf_coh_specs;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%
% for ii = 1:niterations
%     disp(["do nnmf:" ii]);
%     next_nnmf = teh_spectral_nnmf( S );
%     disp('teh_spectral_nnmf done')
%     for jj = 1:ncomps
%         f = fit( linspace(1,nsamples,nsamples)',next_nnmf.nnmf_coh_specs(jj,:)', 'gauss1',options);
%         resid = next_nnmf.nnmf_coh_specs(jj,:) - gauss_func(1:nsamples,f);
%         ss(ii,jj) = sum( resid.^2 );
%     end

%     if sum(ss(ii,:)) < winning_value
%         nnmf_res = next_nnmf;
%         winning_value = sum(ss(ii,:));
%     end

%     if summary_plot == true
%         specs(:,:,ii) = next_nnmf.nnmf_coh_specs;
%     end

% end

%%%%%%%%%%%%%%%%%%%%%%%%%%

if summary_plot
    nrows = ceil( niterations/5 );
    winning_ind = find(winning_value == sum(ss,2));

    figure('Position',[100 100 1536 768])
    for ii = 1:niterations
        subplot( nrows,5, ii);
        plot( specs(:,:,ii)','linewidth',2);grid on;
        title_text = [ num2str(ii) '- SS: ' num2str(sum(ss(ii,:)))];
        if ii == winning_ind
            title_text = [title_text ' - WINNER'];
        end
        title(title_text,'FontSize',14);
    end

    figure
    x_vect = 1:niterations;
    h = bar(x_vect, sum(ss,2) );
    grid on;hold on
    bar(x_vect(winning_ind),sum(ss(winning_ind,:)),'r')
    xlabel('Iteration')
    ylabel('Residual Sum Squares')
    set(gca,'FontSize',14);

end

end
