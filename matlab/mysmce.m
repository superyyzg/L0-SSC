%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
% Changed for RSMG
%--------------------------------------------------------------------------

function [perf,smce_alpha] = mysmce(data,k,tlabel,lambda,KMax,verbose) 

%addpath(fullfile('.','utility','SMCE_v1.2'));
if (nargin < 7)
    verbose = 0;
end

X = data';
% solve the sparse optimization program
[W,smce_alpha] = smce_optimization(X,lambda,KMax,verbose);
W = processC(W,0.95);

% symmetrize the adjacency matrices
Wsym = max(abs(W),abs(W)');

%perf = sc(k,Wsym,tlabel,opt.km_iter);
%perf = usencut(Wsym,k,tlabel);

[clusts,~]=gcut(sparse(Wsym),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel);

end