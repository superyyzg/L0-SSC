%--------------------------------------------------------------------------
% This function computes the maximum error between elements of two 
% coefficient matrices
% C: NxN coefficient matrix
% Z: NxN coefficient matrix
% err: mean absolute error between C and Z
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function err = errorCoef(Z,C)

err = sum(sum( abs(Z-C) )) / (size(C,1)*size(C,2)) ;
