function [perf,alpha] = l1graph_admm(data,k,tlabel,opt,lambda)

% X: d*n data matrix
% lambda ||y - A*x||^2 + lambda*||x||_1

%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%
%lambda = 0.1;
%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

Dist = sqrt(computeDistL2(X'));

alpha = zeros(n);

%matlabpool open;
invAs = cell(1,n);
iter = 1;
lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,'begin l1graph_admm points: ');
XtX = X'*X;
for i = 1:n,
    %idxi = logical(adj(:,i));
    %Xt = X(:,idxi);
    %XtX = Xt'*Xt;
    P = 2*XtX;
    b1 = -2*XtX(:,i);
    b2= ones(n,1)*lambda;
    [outalpha,invAs] = admm_solve_alpha(invAs,i,alpha(:,i),P,b1,b2);
    alpha(:,i) = outalpha;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);
end
textprogressbar(1,lastprintlength,' Done');
%matlabpool close;

W = .5*(abs(alpha)+abs(alpha'));
%addpath(genpath('.\utility'));

%perf = sc(k,W,tlabel,opt.km_iter);
perf = usencut(W,k,tlabel);