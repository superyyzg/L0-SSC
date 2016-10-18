%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Y,Eval] = SpectralEmbedding(W,d)

N = size(W,1);
if (d > N-1)
    d = N-1;
end

D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[~,S,V] = svd(L);
Y = D * V(:,end-1:-1:end-d);
Eval = diag(S(end-1:-1:end-d,end-1:-1:end-d));