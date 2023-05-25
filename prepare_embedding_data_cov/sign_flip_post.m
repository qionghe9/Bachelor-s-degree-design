function [energy, inds]=sign_flip_post(netmat,flips,group_netmat,sigma,num_embeddings,pinv_group_netmat,method,num_lags,inds)
%评估网络结构的差异
energy=1e32;

flips=sparse(diag(kron(flips',ones(1,num_embeddings))));%创建稀疏矩阵 提取主对角线元素 计算Kronecker 积
%将flips扩展为一个大小为num_embeddings的矩阵，对角线上的元素为flips向量的重复。使用稀疏矩阵存储结果，节省内存。
switch method
    case 'matrix_distance_metric'
        [metric, inds]=matrix_distance_metric(flips*netmat*flips,group_netmat,num_lags,'none',inds);
        energy = -metric;
    otherwise
        if sigma==0 %没有噪声
            %Sigmab=flips*group_netmat*flips;
            
            pinv_Sigmab=flips*pinv_group_netmat*flips;
            energy=trace(netmat*pinv_Sigmab);%+logdet(Sigmab); %计算方阵的迹，方阵的迹是指对角线上元素的和

        else %有噪声
            Sigmab=eye(size(netmat))*sigma^2 + flips*group_netmat*flips;%加权和？
            pinv_Sigmab=pinv(Sigmab);
            energy=trace(netmat*pinv_Sigmab)+logdet(Sigmab);
        end;
        
end;

end

