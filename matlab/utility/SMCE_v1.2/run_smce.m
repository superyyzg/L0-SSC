%--------------------------------------------------------------------------
% This is the main function to run SMCE
% -------------------- Input Data and Parameters --------------------------
% Y = DxN matrix of N data points in the D-dimensional space
% lambda = regularization paramter for the Lasso-type optimization program
% KMax = maximum neighborhood size to select the sparse neighbors from
% dim = dimension of the low-dimensional embedding
% n = number of clusters/manifolds
% gtruth = ground-truth vector of memberships of points to the n manifolds, 
% if not available or have just one manifold enter gtruth = []
% verbose = if true, will report the SMCE optimization information
% ------------------------- SMCE Outputs  ---------------------------------
% Yc = has n cells where each cell contains the dim-dimensional embedding
% of the data in each cluster
% Yj = joint embedding of all data points in the (n+1)-dimensional space
% clusters = vector of memberships of data points to n clusters
% missrate = clustering error when n > 1 and gtruth is given
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

clc, clear all, warning off

% enter DxN data matrix Y of N data points in the D-dimensional space
Y = randn(2,10);

% set the parameters of the SMCE algorithm
lambda = 10; KMax = 50; dim = 2;

% enter the number of clusters, if just want embedding enter 1
n = 1;
% if n > 1 and know the clustering ground-truth enter below, otherwise 
% enter empty
gtruth = [];

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE on the data
[Yc,Yj,clusters,missrate] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);
