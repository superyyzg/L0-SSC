function [perf,alpha] = l1graph(data,k,tlabel,lambda)

% X: d*n data matrix
% lambda ||y - A*x||^2 + lambda*||x||_1

%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%
%lambda = 0.1;
%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);
Xinit = zeros(n+d-1,1);
%Xinit = zeros(n-1,1);
talpha = zeros(n+d-1,n);
%talpha = zeros(n-1,n);
parpool;
parfor i = 1:n,
    fprintf('l1graph: process datum %d\n', i);
    Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
    A = [X(:,Xflag) eye(d)];
    %A = [X(:,Xflag)];
    Y = X(:,i);
    Xout = l1ls_featuresign (A, Y, lambda/2, Xinit);
    talpha(:,i) = Xout;
end
delete(gcp);
alpha = zeros(n+d,n);
for i = 1:n,
    alpha(:,i) = [talpha(1:i-1,i);0;talpha(i:end,i)];
end

walpha = alpha(1:n,:);

W = .5*(abs(walpha)+abs(walpha'));
%addpath(genpath('.\utility'));

[clusts,~]=gcut(sparse(W),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel); 

%perf = sc(k,W,tlabel,opt.km_iter);
%perf = usencut(W,k,tlabel);






