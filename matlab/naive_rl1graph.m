function [perf,alpha] = naive_rl1graph(data,k,tlabel,alpha0,knn,lambda,gamma,maxIter,thr)

basic_nargins = 7; 
if (nargin < basic_nargins+1)
    maxIter = 10;
end
if (nargin < basic_nargins+2)
    thr = 1e-5;
end

[n,d] = size(data);
DistL2 = computeDistL2(data);
h_sc = (4/(d+2))^(1/(d+4))*mean(std(data))*n^(-1/(d+4));
%h_sc = 0.05*max(sqrt(DistL2(:)));
%W = exp(-DistL2/(2*(h_sc^2)));
W = ones(n);

X = data';
X = X./repmat(sqrt(sum(X.^2)),d,1);
Dist = sqrt(computeDistL2(X'));
[adjmat,~] = computeAdj(Dist,knn);

%A = [X eye(d)];
A = X;
nA = size(A,2);
alpha = alpha0;
alpha1 = zeros(nA,n);

AtA = A'*A;
AtX = A'*X;


verbose = 0; 
mu = 10;

%invAs = zeros(n,n,n);
%t = gamma*(sum(Wrow)+sum(Wcol));
t = gamma*(2*knn);
P_i = 2*AtA+t*eye(nA);
invA = inv(P_i+mu*eye(nA));

W = sparse(W);
adjmat = sparse(adjmat);
alpha = sparse(alpha);
alpha1 = sparse(alpha1);
for iter = 1:maxIter,    
    lastprintlength = 0;
    lastprintlength = textprogressbar(-1,lastprintlength,'begin naive_rl1graph points: ');
    for i = 1:n,
        [alpha] = admm_alpha_general(AtA,AtX,iter,invA,i,alpha,W,adjmat,lambda,gamma,verbose);
        
        %[alpha] = admm_alpha_symmetric(XtX,XtXUs,XtXS,i,alpha,W,adjmat,Y,beta,lambda,gamma,K,verbose,mu);
        lastprintlength = textprogressbar(0,lastprintlength,i,n);
    end
    textprogressbar(1,lastprintlength,' Done');
    err = errorCoef(alpha1,alpha);
    err = full(err);
    fprintf('naive_rl1graph: error = [%1.1e] iter %4.0f \n', err,iter);
    if iter >= 2 && err < thr, break; end    
    alpha1 = alpha;
end

alpha = full(alpha);

affinity = .5*(abs(alpha(1:n,:))+abs(alpha(1:n,:)'));
%addpath(genpath('.\utility'));

%perf = sc(k,affinity,tlabel,opt.km_iter);

[clusts,~]=gcut(sparse(affinity),k);
label = []; 
for kk = 1:length(clusts),
    label(clusts{kk}) = kk;
end
perf(1) = cluster_accuracy(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand(label,tlabel); 

end