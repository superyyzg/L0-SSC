pathsep = '\';
utilitypath = ['.',pathsep,'utility'];
datapath = ['.',pathsep,'Data'];
addpath(genpath(utilitypath));
addpath(genpath(datapath));

[datas,tlabels] = randdata(20,7);

ntest = length(datas);
perf = cell(1,ntest);
alpha = cell(1,ntest);

for i = 1:ntest,
data = datas{i};
tlabel = tlabels{i};
k = length(unique(tlabel));
opt.km_iter = 30;
opt.km_replica = 10;

%lambda is shared by both l1graph and rl1graph
lambda = 0.1; 
[perf{i}.l1perf,l1_alpha] = l1graph(data,k,tlabel,opt,lambda);
[~,~,smce_label,~] = smce(data',lambda,20,k);
perf{i}.smce(1) = cluster_accuracy(smce_label,tlabel);
perf{i}.smce(2) = nmi(tlabel,smce_label);
perf{i}.kmeans = mykmeans(data,k,tlabel,opt.km_replica,opt.km_iter);
perf{i}.scdefault= runsc(data,k,tlabel,opt.km_iter);
%l1_alpha = zeros(size(data,1));

%compute the pairwise similairty W
issurface = 1;
knn = 5;
adjmat = computeW(data,knn,issurface);
gamma = 0.5;

[perf{i}.naiverl1perf,naiverl1graph_alpha] = naive_rl1graph(data,k,tlabel,l1_alpha,adjmat,opt,lambda,gamma);

fastflag = 0;
[alpha{i}] = admm_rl1graph(data',l1_alpha,adjmat,fastflag,lambda,gamma);
affinity = 0.5*(abs(alpha{i})+abs(alpha{i}'));
perf{i}.rl1graph_perf = sc(k,affinity,tlabel,opt.km_iter);

fprintf('accuracy: l1graph %f, smce %f, kmeans %f, sc %f, naivel1 %f, rl1graph %f \n',perf{i}.l1perf(1),perf{i}.smce(1),perf{i}.kmeans(1),perf{i}.scdefault(1),perf{i}.naiverl1perf(1), perf{i}.rl1graph_perf(1));
fprintf('nmi: l1graph %f, smce %f, kmeans %f, sc %f, naivel1 %f, rl1graph %f \n',perf{i}.l1perf(2),perf{i}.smce(2),perf{i}.kmeans(2),perf{i}.scdefault(2),perf{i}.naiverl1perf(2), perf{i}.rl1graph_perf(2));

end