function [alpha,perf,effective_lambda] = cuda_proximal_l0graph(data,k,tlabel,alpha0,lambda,maxIter,thr)

basic_nargins = 4;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda = 0.1;
end
if (nargin < basic_nargins+2)
    % default error thresholds to stop ADMM
    maxIter = 100;    
end
if (nargin < basic_nargins+3)
    % default error thresholds to stop ADMM
    thr = 1*10^-6; 
end

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

%A = [X eye(d)];
A = X;
nA = size(A,2);
AtA = A'*A;


%XtX = X'*X;

%XtXd  = XtX;
%XtXd(1:n+1:n^2) = 0;
%S = norm(XtXd,'fro');
%[S] = svd(XtX);
[S] = svd(AtA);
S1 = 50;

X = X/sqrt(S1+eps);
A = A/sqrt(S1+eps);
%XtX = X'*X;
AtA = A'*A;
AtX = A'*X;

effective_lambda = (S1+eps)*lambda;

if (effective_lambda >= 1),
    alpha = []; perf = [];
    fprintf('effective_lambda is %.5f > 1, terminating the program now \n', effective_lambda);
    return;
end

err = 10*thr;

iter = 1;
%alpha0 = alpha0(1:n,:);
alpha = alpha0;


X = single(X);
alpha = single(alpha);
A = single(A);
AtA = single(AtA);
AtX = single(AtX);
S1 = single(S1);
thr = single(thr);

save('.\cuda\proximal_l0graph_input','X','alpha','A','AtA','AtX','S1','thr');
systemline = ['cuda-proximal-l0graph.exe ', num2str(lambda), ' ', num2str(maxIter), ' ', num2str(1)];


cd('.\cuda');
fprintf('running cuda-proximal-l0graph\n');
system(systemline);

proximal_l0graph_result = load('proximal_l0graph_result.mat');
alpha = double(proximal_l0graph_result.l0graph_alpha);

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



