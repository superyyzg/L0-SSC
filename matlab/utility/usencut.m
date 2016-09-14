function [perf] = usencut(W,k,tlabel)

addpath(genpath('D:\program\srl1graph\utility\Ncut_9\Ncut_9'));

[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,k);

n = size(W,1);
label = zeros(n,1);
for j=1:k,
    id = find(NcutDiscrete(:,j));
    label(id) = j;
end

ulabel = unique(label);
utlabel = unique(tlabel);
nclass = length(ulabel);

% while nclass ~= length(utlabel) || length(label) ~= length(tlabel),
%     [NcutDiscrete,NcutEigenvectors] =discretisation(NcutEigenvectors);
%     n = size(W,1);
%     label = zeros(n,1);
%     for j=1:k,
%         id = find(NcutDiscrete(:,j));
%         label(id) = j;
%     end
%     ulabel = unique(label);
%     utlabel = unique(tlabel);
%     nclass = length(ulabel);
% end

perf(1) = cluster_accuracy_nosameclass(label,tlabel);
perf(2) = MutualInfo(tlabel,label);
perf(3) = adjrand_nosameclass(label,tlabel);

