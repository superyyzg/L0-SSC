addpath(genpath(fullfile('..\','utility')));

for i = 1:4,
    %%pca
    pca = 1;
    if pca,
        [pc,sdata,latent] = princomp(data{i});
        latent_sum = 0;
        for dim = 1:length(latent),
            latent_sum = latent_sum + latent(dim);
            if latent_sum/sum(latent) >= 0.98,
                break;
            end
        end
        data{i} = sdata(:,1:dim);
    end
end

k = zeros(1,4);

for i = 1:4,
    [data{i},tlabel{i}] = process_data_tlabel(data{i},tlabel{i});
    k(i) = length(unique(tlabel{i}));
end

km_opt.km_iter = 5;
km_opt.km_replica = 2;
%lambda is shared by both l1graph and smce
lambda = 0.1; lambda_l0graph = 0.01; maxIter_l1graph_admm = 10000; maxIter_l0graph = 100; 
smce_kmax=15; omp_Ts = [3 4 5];

%parameters for regularization
reg_weight = 0.1; naive_rl1graph_knn = 5; l0l1_knn = 5;
maxSingleIter_l0l1graph = 30; maxIter_l0l1graph = 5; maxIter_naive_rl1 = 5;

%use cuda?
use_cuda = 1;

test_reg = 0;


perf_multipie = cell(1,4);
l1graph_alpha = cell(1,4);
for i = 1:4,
        
    [perf_multipie{i},l1graph_alpha,l0graph_alpha] = run_perf_comp(data{i},k(i),tlabel{i},km_opt,lambda,lambda_l0graph,maxIter_l1graph_admm,maxIter_l0graph,smce_kmax,omp_Ts,...
                                                    reg_weight,naive_rl1graph_knn,l0l1_knn,smce_kmax,maxSingleIter_l0l1graph,maxIter_l0l1graph,maxIter_naive_rl1,use_cuda,test_reg);

    fprintf('cmu multipie %d cluster result: \n', i);
    perf_multipie{i}
end

%[perf,l1graph_alpha,l0graph_alpha] = run_perf_comp(data,k,tlabel,km_opt,lambda,lambda_l0graph,maxIter_l0graph,smce_kmax,omp_T);
    



