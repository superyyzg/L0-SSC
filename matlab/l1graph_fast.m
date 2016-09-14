function [perf,alpha] = l1graph_fast(data,k,tlabel,opt,lambda,dic_size)

% X: d*n data matrix
% lambda ||y - A*x||^2 + lambda*||x||_1

%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%
%lambda = 0.1;
%%%%%parameter set up%%%%%%%%%%%%%%%%%%%%%%%

X = data';
[d,n] = size(X);
X = X./repmat(sqrt(sum(X.^2)),d,1);

Dist = sqrt(computeDistL2(X'));
dic_size = min(n,dic_size);
adj = computeAdj(Dist,dic_size);

Xinit = zeros(dic_size,1);
alpha = zeros(n);



%matlabpool open;
lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,'begin l1graph_fast points: ');
for i = 1:n,
    idxi = logical(adj(:,i));
    %Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
    A = [X(:,idxi)];
    Y = X(:,i);
    Xout = l1ls_featuresign (A, Y, lambda/2, Xinit);
    alpha(idxi,i) = Xout;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);
end
textprogressbar(1,lastprintlength,' Done');
%matlabpool close;

W = .5*(abs(alpha)+abs(alpha'));
%addpath(genpath('.\utility'));

perf = sc(k,W,tlabel,opt.km_iter);






