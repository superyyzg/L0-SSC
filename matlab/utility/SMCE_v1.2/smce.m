%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Yc,Yj,clusters,missrate,W] = smce(Y,lambda,KMax,dim,n,gtruth,verbose)

if (nargin < 7)
    verbose = true;
end
if (nargin < 6)
    gtruth = [];
end
if (nargin < 5)
    n = 1;
end
if (nargin < 4)
    dim  = 2;
end

% solve the sparse optimization program
W = smce_optimization(Y,lambda,KMax,verbose);
W = processC(W,0.95);

% symmetrize the adjacency matrices
Wsym = max(abs(W),abs(W)');

% perform clustering
[Yj,clusters,missrate] = smce_clustering(Wsym,n,dim,gtruth);

% perform embedding
Yc = smce_embedding(Wsym,clusters,dim);