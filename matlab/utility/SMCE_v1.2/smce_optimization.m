%--------------------------------------------------------------------------
% This function solves the optimization function of SMCE for the given
% data points
% X: DxN matrix of N data points in the D-dimensional space
% lambda: regularization parameter of the SMCE optimization program
% KMax = maximum neighborhood size to select the sparse neighbors from
% verbose: ture if want to see the optimization information, else false
% W: NxN sparse matrix of weights obtained by the SMCE algorithm 
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [W,alpha] = smce_optimization(X,lambda,KMax,verbose)

N = size(X,2);

if (nargin < 2)
    lambda = 10;
end
if (nargin < 3)
    KMax = N - 1;
end

if (KMax > N-1) || (KMax < 1)
    KMax = N-1;
end

X2 = sum(X.^2,1);
Dist = sqrt( repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X) );

W = zeros(N,N); % weight matrix used for clustering and dim. reduction
alpha = zeros(N,N);
% solving the SMCE optimization program for data points
%lastprintlength = 0;
%lastprintlength = textprogressbar(-1,lastprintlength,'beginsmce_optimization points: ');
[~,ids_all] = sort(Dist(:,:),'ascend');
W_cell = cell(1,N);
alpha_cell = cell(1,N);
parpool;
parfor i = 1:N
    fprintf('smce: process datum %d\n', i);
    
    ids = ids_all(1:KMax,i);
    y = X(:,ids(1));
    Y = X(:,ids);
    Y(:,1) = [];
    Y = Y - repmat(y,1,KMax-1);
    v = Dist(ids,i);
    v(1) = [];
    v(v<eps) = eps;
    for j = 1:KMax-1
        Y(:,j) = Y(:,j) ./ v(j);
    end
    
    if (verbose)
        fprintf('Point %4.0f, ',i);
    end    
    
    % solving the Lasso-type optimization using ADMM framework
    c = admm_vec_func(Y,v./sum(v),lambda,verbose);
    
    %W(ids(2:KMax),i) = abs(c./v) / sum(abs(c./v));
    
    %W(ids(2:KMax),i) = abs((c./v) / sum(c./v));
    W_cell{i} = abs((c./v) / sum(c./v));
    %alpha(ids(2:KMax),i) = c;
    alpha_cell{i} = c;
    %lastprintlength = textprogressbar(0,lastprintlength,i,N);
end

for i = 1:N
    ids = ids_all(1:KMax,i);
    W(ids(2:KMax),i) = W_cell{i};
    alpha(ids(2:KMax),i) = alpha_cell{i};
end

delete(gcp);
%textprogressbar(1,lastprintlength,' Done');
