function [perf] = mykmeans(data,k,tlabel,replicate,km_iter)

addpath(genpath('..\utility'));

tperf = zeros(3,km_iter);
perf = zeros(3,1);


for i = 1:km_iter,
    %[label,C] = kmeans(data,k,'Replicates',replicate,'EmptyAction','drop');
    [label, ~] = litekmeans(data, k, 'Start', 'cluster', 'Replicates', replicate);
    tperf(1,i) = cluster_accuracy(label,tlabel);
    tperf(2,i) = MutualInfo(tlabel,label);
    tperf(3,i) = adjrand(label,tlabel);
end

perf(1) = sum(tperf(1,tperf(1,:)~=-inf))/sum(tperf(1,:)~=-inf);
perf(2) = sum(tperf(2,tperf(2,:)~=-inf))/sum(tperf(2,:)~=-inf);
perf(3) = sum(tperf(3,tperf(3,:)~=-inf))/sum(tperf(3,:)~=-inf);
