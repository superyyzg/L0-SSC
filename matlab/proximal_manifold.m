function [alpha,perf,obj] = proximal_manifold(data,k,tlabel,opt,alpha0,lambda,gamma,maxIter,thr)

basic_nargins = 5;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda = 0.1;
end
if (nargin < basic_nargins+2)
    % default rl1graph regularization parameter
    gamma = 0.1;
end
if (nargin < basic_nargins+3)
    % default error thresholds to stop ADMM
    maxIter = 100;    
end
if (nargin < basic_nargins+4)
    % default error thresholds to stop ADMM
    thr = [1*10^-6 1*10^-5 1*10^-5]; 
end

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

XtX = X'*X;
[S] = svd(XtX);

thr1 = thr(1);
err = 10*thr1;

iter = 1;
alpha = alpha0;
perf = [];
obj = [];
while ( iter <= maxIter )
    aat = alpha*alpha';
    ata = alpha'*alpha;
    df = 2*XtX*(alpha-eye(n)) + 2*gamma*( (alpha-eye(n))*aat + ata*(alpha-eye(n)) - alpha*alpha + alpha);
    c = 2*S(1) + 2*gamma;
    alpha_proximal = alpha - 1/c*df;
    
    alpha = max(0,abs(alpha_proximal)-lambda/c).*sign(alpha_proximal);
    alpha (1:n+1:n^2) = 0;
    
    err = errorCoef(alpha,alpha0);
    
    fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
    
    alpha0 = alpha;
       
    cW = .5*(abs(alpha)+abs(alpha'));
    perf  = [perf; usencut(cW,k,tlabel)];
    obj = [obj; compute_obj(X,alpha,lambda,gamma)];
    fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
    
    iter = iter+1;

end

% cW = .5*(abs(alpha)+abs(alpha'));
% perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];

end

function obj = compute_obj(X,alpha,lambda,gamma)
    obj = zeros(1,3);
    obj(1) = norm(X-X*alpha,'fro')^2 + lambda*(sum(abs(alpha(:))));
    obj(2) = gamma*norm(alpha*alpha-alpha,'fro')^2;
    obj(3) = obj(1) + obj(2); 
end



