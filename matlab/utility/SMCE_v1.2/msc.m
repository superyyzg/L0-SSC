% This function gets a coefficient matrix and the indices of the clustering
% of the data points and computed the msc vectors for each cluster
% W: coefficient matrix
% indg: indices of the memberships of the points to the clusters
% msc: each cell of msc is the msc vector corresponding to a cluster
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function mscW = msc(W,indg)

if (nargin < 2)
    indg{1} = 1:size(W,2);
end

n = max(indg);
for i = 1:n
    Wg{i} = W(indg==i,indg==i);
    mscW{i} = medianSort(Wg{i});
end