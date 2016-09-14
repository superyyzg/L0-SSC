function [alpha,perf,effective_lambda] = proximal_l0graph(data,k,tlabel,alpha0,lambda,maxIter,thr)

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
    thr = [1*10^-6 1*10^-5 1*10^-5]; 
end

X = data';
[d,n] = size(X);
%X = X - repmat(mean(X,2),1,n);
X = X./repmat(sqrt(sum(X.^2)),d,1);
%X = X./max(X(:));

A = [X eye(d)];
AtA = A'*A;


%XtX = X'*X;

%XtXd  = XtX;
%XtXd(1:n+1:n^2) = 0;
%S = norm(XtXd,'fro');
%[S] = svd(XtX);
[S] = svd(AtA);
S(1) = 50;

X = X/sqrt(S(1)+eps);
A = A/sqrt(S(1)+eps);
%XtX = X'*X;
AtA = A'*A;
AtX = A'*X;

effective_lambda = (S(1)+eps)*lambda;

if (effective_lambda >= 1),
    alpha = []; perf = [];
    fprintf('effective_lambda is %.5f > 1, terminating the program now \n', effective_lambda);
    return;
end

thr1 = thr(1);
err = 10*thr1;

iter = 1;
%alpha0 = alpha0(1:n,:);
alpha = alpha0;

perf = [];
obj = [];

[obj,l2err,spar_err] = compute_obj_robust(X,A,alpha,lambda);
%fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
fprintf('obj is %.5f, l2err is %.5f, spar_err is %.5f \n', obj,l2err,spar_err);
    
while ( iter <= maxIter )
    %df = 2*XtX*(alpha-eye(n));
    %c = 2;
    %c = 100;
    
    %add robustness to noise and outlier
    df = 2*(AtA*alpha-AtX);
    %c = 2*S(1);
    c = 2;
    
    alpha_proximal = alpha - 1/c*df;
    
    alpha = alpha_proximal;
    %if iter == 1,
    %    alpha(alpha.^2 < 0.01) = 0;
    %else
        alpha(alpha.^2 < 2*lambda/c) = 0;
    %end
    
    %alpha (abs(alpha_proximal) <= eps) = 0;
    %alpha (abs(alpha_proximal) > eps) = alpha_proximal(abs(alpha_proximal) > eps);
    %alpha ((abs(alpha_proximal) > eps) & (alpha_proximal.^2 < 2*lambda/c) ) = 0;
    alpha (1:n+d+1:(n+d)*n-d) = 0;
    %alpha (1:n+1:n^2) = 0;
    
    err = errorCoef(alpha,alpha0);
    
    fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
    
    alpha0 = alpha;
       
    %cW = .5*(abs(alpha)+abs(alpha'));
    %perf  = [perf; usencut(cW,k,tlabel)];
    %perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];
    [obj,l2err,spar_err] = compute_obj_robust(X,A,alpha,lambda);
    %fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
    fprintf('obj is %.5f, l2err is %.5f, spar_err is %.5f \n', obj,l2err,spar_err);
    
    iter = iter+1;

end

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

%perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];

end

function [obj,l2err,spar_err] = compute_obj_robust(X,A,alpha,lambda)
    l2err = norm(X-A*alpha,'fro')^2;
    spar_err = lambda*sum(abs(alpha(:)) > eps);
    obj = norm(X-A*alpha,'fro')^2 + lambda*sum(abs(alpha(:)) > eps);    
end

function [obj,l2err,spar_err] = compute_obj(X,alpha,lambda)
    l2err = norm(X-X*alpha,'fro')^2;
    spar_err = lambda*sum(abs(alpha(:)) > eps);
    obj = norm(X-X*alpha,'fro')^2 + lambda*sum(abs(alpha(:)) > eps); 
end



