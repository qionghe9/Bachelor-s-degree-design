function []=myplot(savebase,psd,coh,channels)
%14    12   543    68    68
n=length(channels);
if n<2
    error('Needs to be larger than 2');
end

%%%%%%

nparcels = 68;
K=12;
f_band=1:size(psd,3);
offdiags = eye(nparcels)==0;
psd_all = zeros(K,length(f_band),nparcels);
coh_all= zeros(K,length(f_band),nparcels,nparcels);

for kk=1:K
    G = squeeze(abs(coh(:,kk,f_band,:,:)));
    P = squeeze(abs(psd(:,kk,f_band,~offdiags)));
    P_off=squeeze(abs(psd(:,kk,f_band,offdiags)));
    coh_all(kk,:,:,:) = mean(G,1);
    coh_ste(kk,:,:,:) = std(G,[],1)./sqrt(length(G));
    psd_all(kk,:,:) = mean(P,1);
    psd_ste(kk,:,:) = std(P,[],1)./sqrt(length(P));
end
colorscheme = set1_cols();
% psd_all=psd_all(:,255:end,:);
% psd_ste=psd_ste(:,255:end,:);
% f_band=1:size(psd_all,2);


% for k=1:K
%     if mod(k,2)==1
%         ls{k} = '-';
%     else
%         ls{k}='--';
%     end
%     shadedErrorBar(0.3*f_band,psd_all(k,:,1),psd_ste(k,:,1),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
%     % statelabels{k} = ['State ',int2str(k)];
%     % h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
% end
% grid on;
% title('PSD per state');
% % plot4paper('Frequency');%X\Y轴加上标签,输入参数小于2,Y轴标签为空
% xlabel('Frequency');
% for k=1:K,h(k).DisplayName=['State ',int2str(k)];end
% leg=legend(h,'Location','EastOutside');
channels=[59,57,61];
% channels=[59,15,57,37,61,17];
% channels= [23,49,57];
% channels= [1,2,3,4,5];
%%%%%%
f_band=1:size(psd_all,2);
for i = channels
    for j = channels 
        ii = find(i==channels); jj = find(j==channels);
        subplot(length(channels),length(channels),(ii-1)*length(channels) + jj)
        if ii==jj
            for k=1:K
                if mod(k,2)==1
                    ls{k} = '-';
                else
                    ls{k}='--';
                end
                shadedErrorBar(0.3*f_band,psd_all(k,:,i),psd_ste(k,:,i),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
                % string_panel = ['(PSD ' num2str(i) ')'];hold off;
                grid on;
            end
        else
            for k=1:K
                if mod(k,2)==1
                    ls{k} = '-';
                else
                    ls{k}='--';
                end
                shadedErrorBar(0.3*f_band,coh_all(k,:,i,j),coh_ste(k,:,i,j),{'LineWidth',2,'LineStyle',ls{k},'Color',colorscheme{k}},1);hold on;
                % string_panel = ['(Coh ' num2str(i) ',' num2str(j) ')'];hold off;
                grid on;
            end
        end
        if ii == 1, title(['Channel ' num2str(j)],'FontSize',8); end
        if ii == length(channels), xlabel('Frequency (Hz)'); end
        if jj == 1, ylabel(['Channel ' num2str(i)],'FontSize',8,'FontWeight','bold'); end
        hold off;
    end
end
% suptitle('Multiple Subplots Title');
% for k=1:K;
%     statelabels{k} = ['State ',int2str(k)];%
%     h(k) = plot(NaN,NaN,'Color',colorscheme{k},'LineWidth',2,'LineStyle',ls{k});
% end
% for k=1:K,h(k).DisplayName=['State ',int2str(k)];end
% leg=legend(h,'Location','EastOutside');

print([savebase '/' num2str(channels, '%d,')],'-dpng')
 