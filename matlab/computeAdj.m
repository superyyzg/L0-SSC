function [adj, adj_knn]= computeAdj(W,knn,order_str)
if (nargin < 3)
    order_str = 'ascend';
end
n = size(W,1);
[~,idx] = sort(W,order_str);
adj = zeros(n);
adj_knn = zeros(knn,n);

for i = 1:n,
    adj(idx(1:knn,i),i) = 1;
    if adj(i,i) == 1,
        adj(i,i) = 0;
        adj(idx(knn+1,i),i) = 1;
    end
    adj_knn(:,i) = find(adj(:,i));
end



