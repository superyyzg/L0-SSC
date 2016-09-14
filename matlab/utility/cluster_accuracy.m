function [accuracy] = cluster_accuracy(label,tlabel)

% build weight matrix of assigning label i to tlabel j
n = length(label);
ulabel = unique(label);
utlabel = unique(tlabel);
nclass = length(ulabel);

if nclass ~= length(utlabel) || length(label) ~= length(tlabel)
    disp('class in label should be the same as that in tlabel');
    accuracy = -inf;
    return;
end

Perf = zeros(nclass);

for i = 1:nclass,
    for j = 1:nclass,
        idx = find(label == ulabel(i));
        tidx = find(tlabel == utlabel(j));
        inter = intersect(idx,tidx);
        lidx = length(idx); linter = length(inter);
        if lidx > linter,
            Perf(i,j) = lidx - linter;
        end
    end
end

[Matching,Cost] = Hungarian(Perf);

accuracy = 1-Cost/n;


