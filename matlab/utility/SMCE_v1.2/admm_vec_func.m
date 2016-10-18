%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points obtained by SMCE
% Y: DxN data matrix of N data points in D-dimensional space
% lambda: regularization parameter of the SMCE optimization program
% verbose: ture if want to see the optimization information, else false
% thr1: stopping threshold for the primal coefficient error ||Z - C||
% thr2: stopping threshold for the primal affine error ||1 - C^T 1||
% thr3: stopping threshold for the dual error || Z(iter) - Z(iter-1) ||
% maxIter: maximum number of iterations of ADMM
% C: NxN sparse coefficient matrix returned by the SMCE algorithm
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function C = admm_vec_func(Y,q,lambda,verbose,mu,thr,maxIter)

if (nargin < 3)
    % default SMCE regularization parameter
    lambda = 10; 
end
if (nargin < 4)
    % ture if want to see the optimization information, else false
    verbose = true;
end
if (nargin < 5)
    % default augmented penalty parameter
    mu = 10;
end
if (nargin < 6)
    % default error thresholds to stop ADMM
    thr = [1*10^-6 1*10^-6 1*10^-5]; 
end
if (nargin < 7)
    % default maximum number of iterations of ADMM
    maxIter = 10000; 
end
% default coefficient error threshold to stop ADMM
thr1 = thr(1);
% default affine constraint error threshold to stop ADMM
thr2 = thr(2);
% default dual error threshold to stop ADMM
thr3 = thr(3);
% default dual threshold to stop ADMM
N = size(Y,2);


A = inv((Y'*Y)+mu*eye(N)+mu);
C = zeros(N,1); Z1 = zeros(N,1);
Lambda = zeros(N,1);
gamma = 0;
err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr2;
i = 1;
% ADMM iterations
while ( (err1(i) > thr1 || err2(i) > thr2 || err3(i) > thr3) && i < maxIter )
    % updating Z
    Z2 = A * (mu*C-Lambda+gamma+mu);
    % updating C
    C = max(0,(abs(mu*Z2+Lambda) - lambda.*q)) .* sign(mu*Z2+Lambda);
    C = 1/mu * C;
    % updating Lagrange multipliers
    Lambda = Lambda + mu * (Z2 - C);
    gamma = gamma + mu * (1 - sum(Z2,1));
    % computing errors
    err1(i+1) = errorCoef(Z2,C);
    err2(i+1) = errorCoef(sum(Z2),1);
    err3(i+1) = errorCoef(Z1,Z2);
    %
    Z1 = Z2;
    i = i + 1;
end

% show the sparse optimization information
if (verbose)
    fprintf('errors = [%1.1e %1.1e %1.1e], iter: %4.0f \n',err1(end),err2(end),err3(end),i);
end
