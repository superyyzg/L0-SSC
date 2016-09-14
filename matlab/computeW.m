function W = computeW(data,knn,issurface)

[n,d] = size(data);

if issurface,
    data = data./repmat(sqrt(sum(data.^2,2)),1,d);
end
%data = data./max(data(:));
DistL2 = computeDistL2(data);
% 
% 
%h_sc = (4/(d+2))^(1/(d+4))*mean(std(data))*n^(-1/(d+4));
%h_sc = 0.05*max(sqrt(DistL2(:)));
%W = exp(-DistL2/(2*(h_sc^2)));

[~,idx] = sort(DistL2);
W = zeros(n);
h = zeros(n,1);
for i = 1:n,
    %h(i) = mean(sqrt(DistL2(idx(2:knn+1,i),i)));
    
    %W(idx(2:knn+1,i),i) = exp(-DistL2(idx(2:knn+1,i),i)./(2*h(i)^2));
    W(idx(2:knn+1,i),i) = 1;
    %W(i,idx(2:knn+1,i)) = 1;
end

% h_sc = mean(h);
% W = exp(-DistL2/(2*(h_sc^2)));





