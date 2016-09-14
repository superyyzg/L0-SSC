function [W,sr,full_diag,Uz] = updateW(alpha,k,initW,Y,gamma,beta,max_W_iter)

W0 = initW;
n = size(W0,1);

%debug info
sr = zeros(max_W_iter,1);
full_diag = zeros(n,max_W_iter);
for iter = 1:max_W_iter,
    
    L = graph_laplacian(W0);
    %[U,~,~] = lansvd((L+1e-6*eye(n)),k,'S');
    [U,S,~] = svd(L);
    Uz = U;
    dS = diag(S);
    sr(iter) = sum(dS(end-k+1:end));
    full_diag(:,iter) = dS;

    g = U(:,end-k+1:end)*diag(ones(k,1))*U(:,end-k+1:end)';
    dg = diag(g);
    wg = (repmat(dg,1,n)+repmat(dg',n,1)-g-g')/2;
    wg(1:(n+1):n^2) = 0;
    
    W = max(0,abs(beta*alpha-Y)-gamma*wg).*sign(beta*alpha-Y)/beta;
    
    err = errorCoef(W,W0);
    
    fprintf('updateW: error = [%1.1e], iter: %4.0f \n',err,iter);
    
    if (err < 1e-6), break; end
    
    W0 = W;
    
end

function [L] = graph_laplacian(W)

affinity = (abs(W) + abs(W'))/2;
L = diag(sum(affinity,2)) - affinity;

