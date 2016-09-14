addpath(genpath(fullfile('..\','utility')));

%load_orl
%knn = 5; gamma = .5
% knn = 10; gamma = 1


%%pca

[pc,sdata,latent] = princomp(data);
latent_sum = 0;
for dim = 1:length(latent),
    latent_sum = latent_sum + latent(dim);
    if latent_sum/sum(latent) >= 0.98,
        break;
    end
end
data = sdata(:,1:dim);

data = orl(1:400,:);
tlabel = orl_tlabel(1:400);



%data = wine;
%tlabel = wine_tlabel;

%knn = 5; gamma = 0.15
%data = yale; 
%tlabel = yale_tlabel;

%knn = 5; gamma = 1.5;
%data = parkinson;
%tlabel = parkinson_tlabel;

%knn = 10; gamma = 3
%data = heart;
%tlabel = heart_tlabel;

%knn = 15; gamma = 5
%data = bt;
%tlabel = bt_tlabel;

%knn = 5; gamma = 0.15
%data = libra;
%tlabel = libra_tlabel;

%knn = 15; gamma = 5
%data = synthetic_control;
%tlabel = synthetic_control_tlabel;

%knn = 100; gamma = 50;
%data = bc;
%tlabel = bc_tlabel;

[data,tlabel] = process_data_tlabel(data,tlabel);
k = length(unique(tlabel));
opt.km_iter = 30;
opt.km_replica = 10;

%lambda is shared by both l1graph and rl1graph
lambda = 0.1;
[perf.l1graph,l1graph_alpha] = l1graph(data,k,tlabel,opt,lambda);
n = size(data,1);
walpha = l1graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.l1graph_ncut = usencut(W,k,tlabel);

T = 3;
[perf.omp,ompl0_alpha] = l0omp(data,k,tlabel,T);
walpha = ompl0_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.omp_ncut = usencut(W,k,tlabel);
lambda_l0graph = 0.01; maxIter_l0graph = 100;
[l0graph_alpha,perf.l0graph,~] = proximal_l0graph(data,k,tlabel,opt,l1graph_alpha,lambda_l0graph,maxIter_l0graph);
walpha = l0graph_alpha(1:n,:);
W = .5*(abs(walpha)+abs(walpha'));
perf.l0graph_ncut = usencut(W,k,tlabel);

dic_size = min(300,size(data,1));
[perf.l1graph_fast,l1graph_fast_alpha] = l1graph_fast(data,k,tlabel,opt,lambda,dic_size);
KMax = 50; %15;
[perf.smce_perf,smce_alpha] = mysmce(data,k,tlabel,opt,lambda,KMax+1);

[perf.l1graph_nonneg,l1graph_nonneg_alpha] = l1graph_nonneg(data,k,tlabel,opt,lambda,dic_size);

gamma = 0.1;
maxIter = 10;
[srcccp_alpha,sr,full_diag,Uz] = srcccp(data,k,tlabel,opt,l1graph_alpha,dic_size,lambda,gamma,maxIter);
W = .5*(abs(srcccp_alpha)+abs(srcccp_alpha'));
%perf.srcccp = sc(k,W,tlabel,opt.km_iter);
perf.srcccp = usencut(W,k,tlabel);


maxIter = 10;
[sradmm_alpha,srW] = admm_srl1graph(data,k,tlabel,opt,l1graph_fast_alpha,dic_size,lambda,gamma,maxIter);
W = .5*(abs(admm_alpha)+abs(admm_alpha'));
perf.sradmm = sc(k,W,tlabel,opt.km_iter);


