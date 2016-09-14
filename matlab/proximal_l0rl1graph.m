function [alpha,perf] = proximal_l0rl1graph(data,k,tlabel,alpha0,lambda_l1,lambda_l0,knn,maxSingleIter,maxIter,verbose,thr)

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
    verbose = false; 
end
if (nargin < basic_nargins+7)
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
AtX = A'*X;

%XtX = X'*X;

%XtXd  = XtX;
%XtXd(1:n+1:n^2) = 0;
%S = norm(XtXd,'fro');
%[S] = svd(XtX);
[S] = svd(AtA);
S(1) = 50;

%X = X/sqrt(S(1)+eps);
%A = A/sqrt(S(1)+eps);
%AtA = A'*A;
%AtX = A'*X;

%effective_lambda = (S(1)+eps)*lambda_l0;

% if (effective_lambda >= 1),
%     alpha = []; perf = [];
%     fprintf('effective_lambda is %.5f > 1, terminating the program now \n', effective_lambda);
%     return;
% end

thr1 = thr(1);
err = 10*thr1;

%alpha0 = alpha0(1:n,:);
alpha = alpha0;

perf = [];
obj = [];

W0 = (abs(alpha0(1:n,:))+abs(alpha0(1:n,:)'))/2;
W0_idx = (W0 > eps);

Dist = sqrt(computeDistL2(X'));
[adjmat,~] = computeAdj(Dist,knn);
adjmat = logical(adjmat);


lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,'begin l0l1graph points: ');

%alpha_neighbors = cell(1,n);
%for i = 1:n,
%    alpha_neighbors{i} = alpha(:,W0_idx(i,:));
%end
%parpool;
tic;
for outer_iter = 1:maxIter,

for i = 1:n,
    
    %alpha_neighbor = alpha(:,W0_idx(i,:));
    %W0_neighbor = W0(i,W0_idx(i,:));
    
    alpha_neighbor = alpha(:,adjmat(:,i));
    W0_neighbor = ones(1,knn);
    
    

    alphai0 = alpha(:,i); 
    alphai = alphai0;
    
    %[obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai0,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
    
    %fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
    
    iter = 1;
    while ( iter <= maxSingleIter )
        %df = 2*XtX*(alpha-eye(n));
        %c = 2;
        %c = 100;

        %add robustness to noise and outlier
        df = 2*(AtA*alphai-AtX(:,i));
        c = 2*S(1);
        %c = 2;

        alpha_proximal = alphai - 1/c*df;

        %alpha(alpha.^2 < 2*lambda/c) = 0;

        [alphai,~] = sol_l0_l1(alpha_proximal,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0);

        alphai(i) = 0;
        %alpha (1:n+d+1:(n+d)*n-d) = 0;
        %alpha (1:n+1:n^2) = 0;

        err = errorCoef(alphai,alphai0);

        

        alphai0 = alphai;

        %cW = .5*(abs(alpha)+abs(alpha'));
        %perf  = [perf; usencut(cW,k,tlabel)];
        %perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];
        [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
        %fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
        if verbose,
            fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
            fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
        end

        iter = iter+1;

    end %while
    
    alpha(:,i) = alphai;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);

end %for

end %outer_iter
toc;

textprogressbar(1,lastprintlength,' Done');

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

function [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0)
    n = size(alphai,1);
    l2err = norm(X(:,i)-A*alphai,'fro')^2;
    l1_spar_err = lambda_l1*sum(abs(alphai));
    
    nn = size(alpha_neighbor,2);
    l0_spar_err = lambda_l0*(abs(repmat(alphai,1,nn) - alpha_neighbor)>eps).*repmat(W0_neighbor,n,1);
    l0_spar_err = sum(l0_spar_err(:));
    obj = l2err + l1_spar_err + l0_spar_err;
end

function [alphai,obj_mat] = sol_l0_l1(alphai_prox,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0)
    
    n = size(alphai_prox,1);
    nn = size(alpha_neighbor,2);
        
    l1_solution = max(abs(alphai_prox) - lambda_l1/c,0).*sign(alphai_prox);
    all_sols = [l1_solution alpha_neighbor];
    
    obj_mat = c/2*(all_sols - repmat(alphai_prox,1,nn+1)).^2 + lambda_l1*(abs(all_sols));
    
    for t = 1:nn,
        obj_mat = obj_mat + lambda_l0*W0_neighbor(t)*(abs(all_sols - repmat(alpha_neighbor(:,t),1,nn+1))>eps);
    end
    
    [~,min_idx] = min(obj_mat,[],2);
    
    alphai = all_sols(sub2ind([n nn+1],(1:n)',min_idx));
end



