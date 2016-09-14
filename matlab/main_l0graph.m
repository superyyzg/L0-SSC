addpath(genpath(fullfile('..\','utility')));

%%pca
if pca,
    [pc,sdata,latent] = princomp(data);
    latent_sum = 0;
    for dim = 1:length(latent),
        latent_sum = latent_sum + latent(dim);
        if latent_sum/sum(latent) >= 0.98,
            break;
        end
    end
    data = sdata(:,1:dim);
end

%yale b K
K = [10 15 20 30 38];

[data,tlabel] = process_data_tlabel(data,tlabel);

k = length(unique(tlabel));
km_opt.km_iter = 30;
km_opt.km_replica = 10;

%lambda is shared by both l1graph and smce
lambda = 0.1; lambda_l0graph = 0.01; maxIter_l0graph = 100;
smce_kmax=15; omp_T = 3;


nK = length(K);
perf = cell(1,nK);
for i = 1:nK,
    k = K(i);
    [data_k_class,tlabel_k_class]  = choose_k_class(data,tlabel,k);
    [perf{i},~] = run_perf_comp(data_k_class,k,tlabel_k_class,km_opt,lambda,lambda_l0graph,maxIter_l0graph,smce_kmax,omp_T);
end

%[perf,l1graph_alpha,l0graph_alpha] = run_perf_comp(data,k,tlabel,km_opt,lambda,lambda_l0graph,maxIter_l0graph,smce_kmax,omp_T);
    



