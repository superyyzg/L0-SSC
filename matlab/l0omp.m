function [perf,ompl0_alpha] = l0omp(data,k,tlabel,T)

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

XtX = X'*X;
talpha = zeros(n-1,n);
parpool;
parfor i = 1:n,
    fprintf('l0omp:process datum %d\n', i);
    Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
    A = [X(:,Xflag)];
    %A = [X(:,Xflag)];
    Y = X(:,i);
    G = XtX;
    G(i,:) = []; G(:,i) = [];
    Xout = omp(A,Y,G,T);
    talpha(:,i) = full(Xout);
end
delete(gcp);
alpha = zeros(n,n);
for i = 1:n,
    alpha(:,i) = [talpha(1:i-1,i);0;talpha(i:end,i)];
end

ompl0_alpha = alpha;

W = .5*(abs(ompl0_alpha)+abs(ompl0_alpha'));
%addpath(genpath('.\utility'));

[clusts,~]=gcut(sparse(W),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel); 
