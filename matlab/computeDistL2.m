function DistL2 = computeDistL2(data)
n = size(data,1);
U = repmat(sum(data.^2,2),1,n);
V = repmat((sum(data.^2,2))',n,1);
DistL2 = U-2*(data*data')+V;
end