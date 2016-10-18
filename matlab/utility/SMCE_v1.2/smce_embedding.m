%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Yg,indg] = smce_embedding(W,grp,dim)

if (nargin < 2)
    grp = ones(1,size(W,1));
end
if (nargin < 3)
    dim = 3 * ones(1,size(W,1));
end

n = max(grp);

% find the embedding for each cluster
for i = 1:n
    indg{i} = find(grp == i);
    Ng(i) = length(indg{i});
    Wg{i} = W(indg{i},indg{i});
    Yg{i} = SpectralEmbedding(Wg{i},dim);
    dim = size(Yg{i},2);
    Yg{i} = Yg{i}(:,1:dim)'*sqrt(Ng(i));
end