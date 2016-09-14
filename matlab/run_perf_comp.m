function [perf,l1graph_alpha,l0graph_alpha] = run_perf_comp(data,k,tlabel,km_opt,lambda,lambda_l0graph,maxIter_l1graph_admm,maxIter_l0graph,smce_kmax,omp_Ts,varargin)

if ~isempty(varargin),
    
    reg_weight = varargin{1};
    naive_rl1graph_knn = varargin{2};
    l0l1_knn = varargin{3};
    smce_kmax = varargin{4};
    maxSingleIter_l0l1graph = varargin{5};
    maxIter_l0l1graph = varargin{6};
    maxIter_naive_rl1 = varargin{7};
    
    % if we should use CUDA version for l1graph,l0graph and l0graph with regularization
    use_cuda = varargin{8};
    
    test_reg = varargin{9};
end

%kmeans
perf.kmeans = mykmeans(data,k,tlabel,km_opt.km_replica,km_opt.km_iter);

%sc
knn_sparse_sc = 0;
[perf.sc,perf.sc_ncut] = runsc(data,k,tlabel,knn_sparse_sc);


if use_cuda,
%cuda_l1graph
[cuda_l1graph_alpha, perf.cuda_l1graph] = cuda_l1graph_admm(data,k,tlabel,lambda,maxIter_l1graph_admm);
n = size(data,1);
walpha = cuda_l1graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.cuda_l1graph_ncut = perf.cuda_l1graph;%usencut(W,k,tlabel);
fprintf('perf.cuda_l1graph = %.5f\n',perf.cuda_l1graph);
fprintf('perf.cuda_l1graph_ncut = %.5f\n',perf.cuda_l1graph_ncut);

%cuda_l0graph
[cuda_l0graph_alpha,perf.cuda_l0graph,~] = cuda_proximal_l0graph(data,k,tlabel,cuda_l1graph_alpha,lambda_l0graph,maxIter_l0graph);
walpha = cuda_l0graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.cuda_l0graph_ncut = perf.cuda_l0graph; %usencut(W,k,tlabel);
fprintf('perf.cuda_l0graph = %.5f\n',perf.cuda_l0graph);
fprintf('perf.cuda_l0graph_ncut = %.5f\n',perf.cuda_l0graph_ncut);

if test_reg,
    %l0-reg-graph
    lambda_l1_for_l0 = 0;
    lambda_l0 = reg_weight;
    %[l0l1graph_alpha,perf.l0l1graph] = proximal_l0rl1graph(data,k,tlabel,l1graph_alpha,lambda_l1,lambda_l0,knn,maxSingleIter_l0graph,maxIter_l0l1graph);
    [cuda_l0l1graph_alpha,perf.cuda_l0l1graph] = cuda_proximal_l0rl1graph(data,k,tlabel,cuda_l0graph_alpha,lambda_l1_for_l0,lambda_l0,l0l1_knn,maxSingleIter_l0l1graph,maxIter_l0l1graph);
    walpha = cuda_l0l1graph_alpha(1:n,:);
    W = .5*(abs(walpha)+abs(walpha'));
    perf.cuda_l0l1graph_ncut = usencut(W,k,tlabel);
    fprintf('perf.cuda_l0l1graph = %.5f\n',perf.cuda_l0l1graph);
    fprintf('perf.cuda_l0l1graph_ncut = %.5f\n',perf.cuda_l0l1graph_ncut);
end

l1graph_alpha = cuda_l1graph_alpha;
l0graph_alpha = cuda_l0graph_alpha;

else
%l1graph
[perf.l1graph,l1graph_alpha] = l1graph(data,k,tlabel,lambda);
n = size(data,1);
walpha = l1graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.l1graph_ncut = usencut(W,k,tlabel);

%l0graph
[l0graph_alpha,perf.l0graph,~] = proximal_l0graph(data,k,tlabel,l1graph_alpha,lambda_l0graph,maxIter_l0graph);
walpha = l0graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.l0graph_ncut = usencut(W,k,tlabel);

end   
    

%smce
%dic_size = min(300,size(data,1));
%[perf.l1graph_fast,l1graph_fast_alpha] = l1graph_fast(data,k,tlabel,opt,lambda,dic_size);
[perf.smce_perf,smce_alpha] = mysmce(data,k,tlabel,lambda,smce_kmax+1);
walpha = smce_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.smce_ncut = perf.smce_perf;%usencut(W,k,tlabel);

%omp
perf_omp = zeros(length(omp_Ts),3);
perf_omp_ncut = zeros(length(omp_Ts),3);
for i = 1:length(omp_Ts),
    [perf_omp(i,:),ompl0_alpha] = l0omp(data,k,tlabel,omp_Ts(i));
    walpha = ompl0_alpha(1:n,:);
    W = .5*(abs(walpha)+abs(walpha'));
    perf_omp_ncut(i,:) = perf_omp(i,:); %usencut(W,k,tlabel);
end
if length(omp_Ts) > 1,
    perf.omp = max(perf_omp);
    perf.omp_ncut = max(perf_omp_ncut);
else
    perf.omp = perf_omp;
    perf.omp_ncut = perf_omp_ncut;
end








if test_reg,
    %naive_rl1graph
    %the regularization weight of naive rl1 is twice that of l0l1
    gamma = 2*reg_weight;
    [perf.naiverl1graph,naiverl1graph_alpha] = naive_rl1graph(data,k,tlabel,l1graph_alpha,naive_rl1graph_knn,lambda,gamma,maxIter_naive_rl1);
    walpha = naiverl1graph_alpha(1:n,:);
    W = .5*(abs(walpha)+abs(walpha'));
    perf.naiverl1graph_ncut = usencut(W,k,tlabel);
    fprintf('perf.naiverl1graph = %.5f\n',perf.naiverl1graph);
    fprintf('perf.naiverl1graph_ncut = %.5f\n',perf.naiverl1graph_ncut);

    
end