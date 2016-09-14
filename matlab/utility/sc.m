% Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In T. Dietterich,
% S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 
% (pp. 849  856). MIT Press.

% Asad Ali
% GIK Institute of Engineering Sciences & Technology, Pakistan
% Email: asad_82@yahoo.com

% CONCEPT: Introduced the normalization process of affinity matrix(D-1/2 A D-1/2), 
% eigenvectors orthonormal conversion and clustering by kmeans

function varargout = sc(k,W,varargin)

% cluster the data using the normalized symmetric Laplacian
N = size(W,1);
W = sparse(W);
D = diag( 1./sqrt(sum(W,1)+eps) );
D = sparse(D);
L = sparse(eye(N)) - D * W * D;
%eigs_opt.tol = 1e-3;
%[V,~] = eigs(L,k,'lm',eigs_opt);
%[V,~] = eigs(L,k,'sm');
[V,~,~] = svd(full(L));
% D = diag( 1./sqrt(sum(W,1)+eps) );
% L = eye(N) - D * W * D;
% [~,~,V] = svd(L,'econ');

U = V(:,end:-1:end-k+1);
normU = sqrt(sum(U.^2,2));
U = U ./ (repmat(normU,1,k)+eps);

km_replicate = 20;

if numel(varargin) == 0,
    % perform kmeans clustering on the matrix U
    [label,~] = litekmeans(U, k,'Replicates', km_replicate);
    %[label,~] = kmeans(U,k,'Replicates',km_replicate,'EmptyAction','drop'); 
    varargout{1} = label;
    varargout{2} = U;
elseif  numel(varargin) == 1,
    [label,~] = litekmeans(U, k,'Replicates', km_replicate);
    %[label,C] = kmeans(U,k,'Replicates',km_replicate,'EmptyAction','drop'); 
    tlabel = varargin{1};
    varargout{1} = label';
    perf = zeros(3,1);
    perf(1) = cluster_accuracy(label,tlabel);
    perf(2) = MutualInfo(tlabel,label);
    perf(3) = adjrand(label,tlabel);
    varargout{2} = perf;
elseif numel(varargin) == 2,    
    tlabel = varargin{1};
    km_iter = varargin{2};
    tperf = zeros(3,km_iter);
    perf = zeros(3,1);
    for i = 1:km_iter,
        %[label,~] = litekmeans(U, k,'Replicates', km_replicate);
        [label,~] = kmeans(U,k,'Replicates',10,'EmptyAction','drop');
        tperf(1,i) = cluster_accuracy(label,tlabel);
        tperf(2,i) = MutualInfo(tlabel,label);
        tperf(3,i) = adjrand(label,tlabel);            
    end
    perf(1) = sum(tperf(1,tperf(1,:)~=-inf))/sum(tperf(1,:)~=-inf);
    perf(2) = sum(tperf(2,tperf(2,:)~=-inf))/sum(tperf(2,:)~=-inf);
    perf(3) = sum(tperf(3,tperf(3,:)~=-inf))/sum(tperf(3,:)~=-inf);
    varargout{1} = perf;
elseif  numel(varargin) == 3 && strcmp(varargin{3},'margin')
    tlabel = varargin{1};
    km_iter = varargin{2};
    tperf = zeros(3,km_iter);
    perf = zeros(3,1);
    margin = zeros(km_iter,1);    
    for i = 1:km_iter,
        [label,~] = litekmeans(U, k,'Replicates', km_replicate);
        %[label,C] = kmeans(U,k,'Replicates',100,'EmptyAction','drop');
        tperf(1,i) = cluster_accuracy(label,tlabel);
        tperf(2,i) = MutualInfo(tlabel,label);
        tperf(3,i) = adjrand(label,tlabel);
        margin(i) =  compute_margin(k,label,affinity);
    end
   
    perf(1) = sum(tperf(1,tperf(1,:)~=-inf))/sum(tperf(1,:)~=-inf);
    perf(2) = sum(tperf(2,tperf(2,:)~=-inf))/sum(tperf(2,:)~=-inf);
    perf(3) = sum(tperf(3,tperf(3,:)~=-inf))/sum(tperf(3,:)~=-inf);
    varargout{1} = perf;
    varargout{1} = sum(perf(perf~=-inf),2)/sum(perf~=-inf,2);
    varargout{2} = sum(margin(margin~=-inf))/sum(margin~=-inf);
end

function margin = compute_margin(k,label,affinity)

ulabel = unique(label);
if length(ulabel) ~= k,
    margin = -inf;
    return;
end

n = length(label);
margin = 0;

for i = 1:n,
    margin = margin + sum(affinity(i,label~=label(i)));
end

% for c = 1:k,
%     idxc = find(label==ulabel(c)); idxnc = find(label~=ulabel(c));
%     volc = sum(sum(affinity(idxc,idxc)))-sum(diag(affinity(idxc,idxc))); cutc = sum(sum(affinity(idxc,idxnc)));
%     margin = margin + cutc/volc;
% end
        

    


% plot the eigen vector corresponding to the largest eigen value
%figure,plot(IDX)
% figure,
% hold on;
% for i=1:size(IDX,1)
%     if IDX(i,1) == 1
%         plot(data(i,1),data(i,2),'m+');
%     elseif IDX(i,1) == 2
%         plot(data(i,1),data(i,2),'g+');
%     else
%         plot(data(i,1),data(i,2),'b+');        
%     end
% end
% hold off;
% title('Clustering Results using K-means');
% grid on;shg