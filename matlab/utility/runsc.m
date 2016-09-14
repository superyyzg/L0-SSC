function [perf,perf_ncut] = runsc(data,k,tlabel,knn_sparse)

if nargin < 4,
    knn_sparse = 0;
end

[n,d] = size(data);
DistL2 = computeDistL2(data);
%h_sc = (4/(d+2))^(1/(d+4))*mean(std(data))*n^(-1/(d+4));
%h_sc = 0.05*max(sqrt(DistL2(:)));
[~,idx_dist] = sort(DistL2);
idx_dd = find(idx_dist == repmat(1:n,n,1));
idx_dist(idx_dd) = [];
idx_dist = reshape(idx_dist,n-1,n);

self_tuned_K = min(7,n-1);
sigma = sqrt(DistL2(sub2ind([n,n], idx_dist(self_tuned_K,:), 1:n)));
W = exp(-DistL2./(sigma'*sigma));

%sparse W by KNN
if knn_sparse,
    [~,idx] = sort(DistL2);
    adj = zeros(n);
    for i = 1:n,
        adj(idx(1:knn_sparse,i),i) = 1;
        if adj(i,i) == 1,
            adj(i,i) = 0;
            adj(idx(knn_sparse+1,i),i) = 1;
        end        
    end
    adj = adj & adj';
    W = W .* adj;
end

%W = exp(-DistL2/(2*(h_sc^2)));

[clusts,~]=gcut(sparse(W),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel);

perf_ncut = perf;%usencut(W,k,tlabel);