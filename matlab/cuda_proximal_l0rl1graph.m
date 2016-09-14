function [alpha,perf] = cuda_proximal_l0rl1graph(data,k,tlabel,alpha0,lambda_l1,lambda_l0,knn,maxSingleIter,maxIter,thr)

basic_nargins = 4;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda_l1 = 0.1;
end
if (nargin < basic_nargins+2)
    % default rl1graph regularization parameter
    lambda_l0 = 0.01;
end
if (nargin < basic_nargins+3)
    % default error thresholds to stop ADMM
    knn = 5;    
end
if (nargin < basic_nargins+4)
    % default error thresholds to stop ADMM
    maxSingleIter = 100;    
end
if (nargin < basic_nargins+5)
    % default error thresholds to stop ADMM
    maxIter = 5;    
end
if (nargin < basic_nargins+6)
    % default error thresholds to stop ADMM
    thr = [1*10^-6 1*10^-5 1*10^-5]; 
end

X = data';
[d,n] = size(X);
%X = X - repmat(mean(X,2),1,n);
X = X./repmat(sqrt(sum(X.^2)),d,1);
%X = X./max(X(:));

%A = [X eye(d)];
A = X;
nA = size(A,2);
AtA = A'*A;
AtX = A'*X;

%XtX = X'*X;

%XtXd  = XtX;
%XtXd(1:n+1:n^2) = 0;
%S = norm(XtXd,'fro');
%[S] = svd(XtX);
[S] = svd(AtA);
S1 = 50;

thr1 = thr(1);
err = 10*thr1;

%alpha0 = alpha0(1:n,:);
alpha = alpha0;

perf = [];
obj = [];

%W0 = (abs(alpha0(1:n,:))+abs(alpha0(1:n,:)'))/2;
%W0_idx = (W0 > eps);

Dist = sqrt(computeDistL2(X'));
%[~, adjmat] = computeAdj(Dist,knn);
[adj,~] = computeAdj(Dist,knn);
adj = adj + adj';
adjmat_size = 0;
for i = 1:n,
    adj_i = find(adj(:,i));
    if length(adj_i) > adjmat_size,
        adjmat_size = length(adj_i);
    end    
end
adjmat = -1*ones(adjmat_size,n);
adjweight = -1*ones(adjmat_size,n);
for i = 1:n,
    adj_i = find(adj(:,i));
    adjmat(1:length(adj_i),i) = adj_i;
    adjweight(1:length(adj_i),i) = adj(adj_i,i);
end


X = single(X);
l1graph_alpha = single(alpha0);
adjmat = single(adjmat);
adjweight = single(adjweight);
A = single(A);
AtA = single(AtA);
AtX = single(AtX);
S1 = single(S1);
knn = single(knn);

%l1graph_alpha = alpha0;

save('.\cuda\l0l1graph_input','X','l1graph_alpha','adjmat', 'adjweight', 'A','AtA','AtX', 'S1', 'knn');
systemline = ['cuda-l0rl1graph.exe ', num2str(lambda_l1), ' ', num2str(lambda_l0), ' ', num2str(maxSingleIter), ' ', num2str(maxIter)];


cd('.\cuda');
fprintf('running cuda-l0l1graph\n');
system(systemline);

l0l1graph_result = load('l0l1graph_result.mat');
alpha = double(l0l1graph_result.l0l1graph_alpha);

cd('..');

walpha = alpha(1:n,:);
cW = .5*(abs(walpha)+abs(walpha'));

[clusts,~]=gcut(sparse(cW),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel);




