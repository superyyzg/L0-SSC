function [perf,alpha] = sl1graph(data,k,tlabel,opt)

lambda = 0.1:0.2:1;
nl = length(lambda);

perf = zeros(3,nl);
alpha = cell(1,nl);

parfor i = 1:nl,
    [perf(:,i),alpha{i}] = l1graph(data,k,tlabel,opt,lambda(i));
end

