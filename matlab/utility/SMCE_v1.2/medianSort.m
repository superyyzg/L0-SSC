% This function gets a similarty matrix C and sort the elements of each
% column from largest to smallest in a new matrix W, finally average these
% sorted columns to get a mean vector v which indicates the histogram of
% the magnitude of the columns of C.
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function v = medianSort(C)

N = size(C,1);
W = zeros(N,N);
for i = 1:N
    [~,ind] = sort(C(:,i),'descend');
    W(:,i) = C(ind,i);
end
v = median(W,2);