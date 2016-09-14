function [alpha,perf] = cuda_l1graph_admm(data,k,tlabel,lambda,maxIter,mu,thr)

% X: d*n data matrix
% lambda ||y - A*x||^2 + lambda*||x||_1

%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%
%lambda = 0.1;
%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%

basic_nargins = 3;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda = 0.1;
end
if (nargin < basic_nargins+2)
    % default rl1graph regularization parameter
    maxIter = 10000;
end
if (nargin < basic_nargins+3)
    % default rl1graph regularization parameter
    mu = 10;
end
if (nargin < basic_nargins+4)
    % default rl1graph regularization parameter
    thr = [1e-9 1e-9];
end

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

%A = [X eye(d)];
A = X;
nA = size(A,2);
AtA = A'*A;
AtX = A'*X;
alpha = zeros(nA,n);

inA = inv(2*AtA+mu*eye(nA));

X = single(X);
alpha = single(alpha);
A = single(A);
AtA = single(AtA);
AtX = single(AtX);
inA = single(inA);
mu = single(mu);
thr = single(thr);

save('.\cuda\l1graph_admm_input','X','alpha','A','AtA','AtX','inA','mu','thr');
systemline = ['cuda-l1graph-admm.exe ', num2str(lambda), ' ', num2str(maxIter)];


cd('.\cuda');
fprintf('running cuda-l1graph-admm\n');
system(systemline);

l1graph_admm_result = load('l1graph_admm_result.mat');
alpha = double(l1graph_admm_result.l1graph_alpha);

cd('..');

walpha = alpha(1:n,:);
cW = .5*(abs(walpha)+abs(walpha'));

%perf = sc(k,cW,tlabel,10);

[clusts,~]=gcut(sparse(cW),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel);
