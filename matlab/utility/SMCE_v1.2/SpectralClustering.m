%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Y,Eval,Grp] = SpectralClustering(W,d,n,MAXiter,REPlic)

N = size(W,1);

D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[~,S,V] = svd(L);
Y = V(:,end-d+1:end);
for i = 1:N
    Y(i,:) = Y(i,:) ./ norm(Y(i,:)+eps);
end
Eval = diag(S(end-n+1:end,end-n+1:end));

if n > 1
    Grp = kmeans(Y(:,end-n+1:end),n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
else 
    Grp = ones(1,N);
end
Y = Y';