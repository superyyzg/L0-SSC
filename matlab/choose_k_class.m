function [data_k_class,tlabel_k_class]  = choose_k_class(data,tlabel,k)
% choose the first 1..k classes from data; tlabel must be 1,2,...,K with k <= K

k_class_idx = [];
for i = 1:k,
    k_class_idx = [k_class_idx; find(tlabel==i)];
end

data_k_class = data(k_class_idx,:);
tlabel_k_class = tlabel(k_class_idx);

