function [accuracy] = cluster_accuracy_nosameclass(label,tlabel)

res = bestMap(tlabel,label);
%=============  evaluate AC: accuracy ==============
accuracy = length(find(tlabel(:) == res(:)))/length(tlabel);