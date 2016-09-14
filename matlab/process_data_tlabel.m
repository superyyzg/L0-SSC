function [data,tlabel] = process_data_tlabel(data,tlabel)

utlabel = unique(tlabel);

k = length(utlabel);

label = zeros(size(tlabel));

for i = 1:k,
    label(tlabel==utlabel(i)) = i;
end

tlabel = label;

