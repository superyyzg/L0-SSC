function [alpha,s,full_diag,Uz] = srcccp(data,k,tlabel,opt,alpha0,dic_size,lambda,gamma,maxIter,thr)
%  data: n by d
%     k: the number of desired clusters
%alpha0: the initial sparse codes
%dic_size: the size of dictionary for srl1graph
%lambda: weight for sparsity of srl1graph
% gamma: the weight of the spectrum regularization (the top k largest eigenvalue of the graph laplacian)
% maxIter: the maximum number of iterations for CCCP

alpha = alpha0;

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

XtX = X'*X;

Dist = sqrt(computeDistL2(X'));
adj = computeAdj(Dist,dic_size);
%adj = ones(n);
%adj(1:n+1:n^2) = 0;
%dic_size = n-1;

s1 = [];
s2 = [];
full_diag = [];
perf = [];
invAs = cell(1,n);
for iter = 1:maxIter,
    
    [D,L] = graph_laplacian(alpha);
    %[U,~,~] = lansvd((L+1e-6*eye(n)),k,'S');
    [U,S,~] = svd(L);
    Uz = U;
    dS = diag(S);
    s1 = [s1; trace(alpha'*XtX*alpha) - 2*trace(XtX*alpha) + lambda*sum(abs(alpha(:)))];
    s2 = [s2; gamma*sum(dS(end-k+1:end))];
    full_diag =  [ full_diag dS];

    g = U(:,end-k+1:end)*diag(-ones(k,1))*U(:,end-k+1:end)';
    g = D*g*D;
    dg = diag(g);
    wg = (repmat(dg,1,n)+repmat(dg',n,1)-g-g')/2;
    
    lastprintlength = 0;
    lastprintlength = textprogressbar(-1,lastprintlength,['begin srcccp iteration ',num2str(iter),':']);
    for i = 1:n,
        %idxi = logical(adj(:,i));
        %Xt = X(:,idxi);
        %XtX = Xt'*Xt;
        P = 2*XtX;
        b1 = -2*XtX(:,i);
        b2 = ones(n,1)*lambda- gamma*wg(:,i);
        [outalpha,invAs] = admm_solve_alpha(invAs,i,alpha(:,i),P,b1,b2);
        alpha(:,i) = outalpha;
        lastprintlength = textprogressbar(0,lastprintlength,i,n);
    end
    textprogressbar(1,lastprintlength,' Done\n');
    if iter > 1,
        err = errorCoef(alpha,alpha0);
    else
        err = 10*thr;
    end
    
    fprintf('srcccp: errors = [%1.1e], iter: %4.0f \n',err,iter);
    
    if err < thr, break; end
     
    alpha0 = alpha;
    
    W = .5*(abs(alpha)+abs(alpha'));
    %perf.srcccp = sc(k,W,tlabel,opt.km_iter);
    %perf = [perf; usencut(W,k,tlabel)];
end

s = [s1 s2];

[D,L] = graph_laplacian(alpha);
Z = alpha;
[U,S,V] = svd(Z,'econ');
S = diag(S);
r = sum(S>1e-4*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^4;

% spectral clustering
D = diag(1./sqrt(sum(L,2)));
L = D*L*D;
[U,S,V] = svd(L);
V = U(:,1:k);
V = D*V;

n = size(V,1);
M = zeros(k,k,20);
rand('state',123456789);
for i=1:size(M,3)
    inds = false(n,1);
    while sum(inds)<k
        j = ceil(rand()*n);
        inds(j) = true;
    end
    M(:,:,i) = V(inds,:);
end

idx = kmeans(V,k,'emptyaction','singleton','start',M,'display','off');


% Uzs = Uz(:,end-k+1:end);
% % compute discretize Ncut vectors
% [NcutDiscrete,NcutEigenvectors] =discretisation(Uzs);
% NcutDiscrete = full(NcutDiscrete);
% 
% label = zeros(n,1);
% for j=1:k,
%     id = find(NcutDiscrete(:,j));
%     label(id) = j;
% end
% 
% perf(1) = cluster_accuracy(label,tlabel);
% perf(2) = MutualInfo(tlabel,label);
% perf(3) = adjrand(label,tlabel);



function [D,L] = graph_laplacian(alpha)

W = (abs(alpha) + abs(alpha'))/2;
D = diag(1./sqrt(sum(W)));
L = D*(diag(sum(W,2)) - W)*D;



