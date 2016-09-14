function W = computeW_manifold(data,knn,issurface,l1_alpha)
[n,d] = size(data);
[idx] = l1_alpha > 1e-8;
W = zeros(n);
for i = 1:n,
    
    W(idx(:,i),i) = 1;
    W(i,i) = 0;
    %W(i,idx(2:knn+1,i)) = 1;
end





