% Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In T. Dietterich,
% S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 
% (pp. 849  856). MIT Press.

% Asad Ali
% GIK Institute of Engineering Sciences & Technology, Pakistan
function varargout = sc1(k,W,varargin)

% function [Eigenvectors,Eigenvalues] = ncut(W,nbEigenValues,dataNcut);
% 
% Input:
%     W= symmetric similarity matrix
%     nbEigenValues=  number of Ncut eigenvectors computed
%     dataNcut= optional parameters
%
%     default parameters for dataNcut:
%     dataNcut.offset = 5e-1; offset in the diagonal of W
%     dataNcut.verbose = 0; 0 for verbose off mode, 1,2,3 for verbose on modes
%     dataNcut.maxiterations = 100; max number of iterations in eigensolver
%     dataNcut.eigsErrorTolerance = 1e-6; error tolerance in eigensolver
%     dataNcut.valeurMin=1e-6; % truncates any values in W less than valeurMin
% 
% Output: 
%    Eigenvectors= continuouse Ncut eigenvectos, size = length(W) x nbEigenValues
%    Eigenvalues= Ncut eigenvalues, size = 1x nbEigenValues
%
% Timothee Cour, Stella Yu, Jianbo Shi, 2004.


dataNcut.offset = 5e-1;
dataNcut.verbose = 0;
dataNcut.maxiterations = 300;
dataNcut.eigsErrorTolerance = 1e-8;
dataNcut.valeurMin=1e-6;

% if nargin < 3
%     dataNcut.offset = 5e-1;
%     dataNcut.verbose = 0;
%     dataNcut.maxiterations = 100;
%     dataNcut.eigsErrorTolerance = 1e-6;
%     dataNcut.valeurMin=1e-6;
% end

% make W matrix sparse
W = sparsifyc(W,dataNcut.valeurMin);
nbEigenValues = k;
% check for matrix symmetry
if max(max(abs(W-W'))) > 1e-10 %voir (-12) 
    %disp(max(max(abs(W-W'))));
    error('W not symmetric');
end

n = size(W,1);
nbEigenValues = min(nbEigenValues,n);
offset = dataNcut.offset;


% degrees and regularization
d = sum(abs(W),2);
dr = 0.5 * (d - sum(W,2));
d = d + offset * 2;
dr = dr + offset;
W = W + spdiags(dr,0,n,n);

Dinvsqrt = 1./sqrt(d+eps);
P = spmtimesd(W,Dinvsqrt,Dinvsqrt);
clear W;

options.issym = 1;
     
if dataNcut.verbose
    options.disp = 3; 
else
    options.disp = 0; 
end
options.maxit = dataNcut.maxiterations;
options.tol = dataNcut.eigsErrorTolerance;

options.v0 = ones(size(P,1),1);
options.p = max(35,2*nbEigenValues); %voir
options.p = min(options.p,n);

%warning off
% [vbar,s,convergence] = eigs2(@mex_w_times_x_symmetric,size(P,1),nbEigenValues,'LA',options,tril(P)); 
[vbar,s,convergence] = eigs(@mex_w_times_x_symmetric,size(P,1),nbEigenValues,'LA',options,tril(P)); 
%warning on

s = real(diag(s));
[x,y] = sort(-s); 
Eigenvalues = -x;
vbar = vbar(:,y);
U = spdiags(Dinvsqrt,0,n,n) * vbar;

km_replicate = 10;

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
        [label,~] = litekmeans(U, k,'Replicates', km_replicate);
        %[label,C] = kmeans(U,k,'Replicates',100,'EmptyAction','drop');
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