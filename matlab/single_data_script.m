addpath(genpath(fullfile('..\','utility')));
data = heart;
tlabel = heart_tlabel;
[data,tlabel] = process_data_tlabel(data,tlabel);
k = length(unique(tlabel));

km_opt.km_iter = 30;
km_opt.km_replica = 10;

lambda = 0.1; lambda_l0graph = 0.01; maxIter_l1graph_admm = 7000; maxIter_l0graph = 100; 
smce_kmax=15; omp_Ts = [3 4 5];

%parameters for regularization
reg_weight = 0.1; naive_rl1graph_knn = 5; l0l1_knn = 5;
maxSingleIter_l0l1graph = 30; maxIter_l0l1graph = 5; maxIter_naive_rl1 = 5;

%use cuda?
use_cuda = 1;

test_reg = 0;

[perf,l1graph_alpha,l0graph_alpha] = run_perf_comp(data,k,tlabel,km_opt,lambda,lambda_l0graph,maxIter_l1graph_admm,maxIter_l0graph,smce_kmax,omp_Ts,...
                                                    reg_weight,naive_rl1graph_knn,l0l1_knn,smce_kmax,maxSingleIter_l0l1graph,maxIter_l0l1graph,maxIter_naive_rl1,use_cuda,test_reg);
mean(sum(abs(l0graph_alpha)>0))
lambda_l0graph = 0.002;
n = size(data,1);
%l0graph
[l0graph_alpha,perf.l0graph,~] = proximal_l0graph(data,k,tlabel,l1graph_alpha,lambda_l0graph,maxIter_l0graph);
walpha = l0graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.l0graph_ncut = usencut(W,k,tlabel);



data = ionosphere; %defaualt lambda_l0graph

data = mnist;
lambda_l0graph = 0.002;