function [ ret ] = subplot_grid( nrow,ncols,r,c )

ret=subplot(nrow,ncols,(r-1)*ncols+c);

end
%生成子图句柄
