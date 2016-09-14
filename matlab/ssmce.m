function [perf] = ssmce(data,tlabel)

lambda = 0.1:0.2:1;
%lambda = 0.1;
nl = length(lambda);

perf = zeros(3,nl);
k = length(unique(tlabel));
for i = 1:nl,
    [~,~,smce_label,~] = smce(data',lambda(i),20,k);
    perf(1,i) = cluster_accuracy(smce_label,tlabel);
    perf(2,i) = MutualInfo(tlabel,smce_label);
    perf(3,i) = adjrand(smce_label,tlabel);    
end

