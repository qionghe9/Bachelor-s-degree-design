function [ metric, inds ] = matrix_distance_metric( x,y,diag_offset,mode, inds )
% assumes symmetric matrix

switch mode
    case 'abs'
        x=abs(x);
        y=abs(y);
    case 'sign'
        x=sign(x);%返回元素符号
        y=sign(y);
end;

if isempty(inds)
    tmp = triu(x,diag_offset);%表示提取矩阵的上三角，diag_offset表示对角线偏移量，例如-1表示主对角线以下一格的上三角部分
    inds = tmp~=0;%选出tmp矩阵不为0元素的坐标
end;

a=x(inds);
b=y(inds);

metric = abs(corrcoef(a,b));%计算a b的相关系数代替距离，返回2X2矩阵
metric=metric(1,2);

end

