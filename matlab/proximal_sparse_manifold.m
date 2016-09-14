function [alpha,perf,obj] = proximal_sparse_manifold(data,k,tlabel,opt,KMax,alpha0,lambda,gamma,maxIter,thr)

basic_nargins = 6;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda = 0.1;
end
if (nargin < basic_nargins+2)
    % default rl1graph regularization parameter
    gamma = 0.1;
end
if (nargin < basic_nargins+3)
    % default error thresholds to stop ADMM
    maxIter = 100;    
end
if (nargin < basic_nargins+4)
    % default error thresholds to stop ADMM
    thr = [1*10^-6 1*10^-5 1*10^-5]; 
end

X = data';
[d,n] = size(X);
%X = X./repmat(sqrt(sum(X.^2)),d,1);
Dist = sqrt(computeDistL2(X'));
%Dist(Dist < 1e-6) = 1e-6;
invD = 1./(Dist+eps);

s = KMax;
A = computeAdj(Dist,s);

B = compute_B(X,A,invD,s);
Q = computeQ(Dist,A);

%XtX = X'*X;
[S] = svd(B);

thr1 = thr(1);
err = 10*thr1;

iter = 1;


alpha = alpha0;
talpha_numerator = alpha.*invD;
Rtalpha = repmat(sum(talpha_numerator),n,1);
talpha = talpha_numerator ./ Rtalpha;
    
perf = [];
obj = [];
while ( iter <= maxIter )
    aat = talpha*talpha';
    ata = talpha'*talpha;
    
    dtalpha = (talpha-eye(n))*aat + ata*(talpha-eye(n)) - talpha*talpha + talpha;
    d_talpha_alpha_t1 = -talpha_numerator./(Rtalpha.^2);
    dalpha = 1./Rtalpha.*dtalpha + repmat(sum(dtalpha.*d_talpha_alpha_t1),n,1);
    dalpha = dalpha.*invD;
    
    ralpah = reshape(alpha(logical(A(:))),s,n);
    
    df = B*ralpah + 2*gamma*reshape(dalpha(logical(A(:))),s,n);
    t = 2*S(1) + 2*gamma;
    alpha_proximal = ralpah - 1/t*df;
    
    
    ralpha  = update_alpha_smc(alpha_proximal,A,Q,ralpah,s,t,lambda);
    
    alpha = zeros(n);
    alpha(logical(A(:))) = ralpha(:);
    %alpha = max(0,abs(alpha_proximal)-lambda/c).*sign(alpha_proximal);
    %alpha (1:n+1:n^2) = 0;
    
    err = errorCoef(alpha,alpha0);
    
    fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
    
    alpha0 = alpha;
    
    talpha_numerator = alpha.*invD;
    Rtalpha = repmat(sum(talpha_numerator),n,1);
    talpha = talpha_numerator ./ Rtalpha;
       
    cW = .5*(abs(talpha));
    cW = processC(cW,0.95);
    cW = max(abs(cW),abs(cW'));
    perf  = [perf; usencut(cW,k,tlabel)];
    obj = [obj; compute_obj(B,A,alpha,invD,lambda,Q,gamma,s)];
    fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
    
    iter = iter+1;

end

% cW = .5*(abs(alpha)+abs(alpha'));
% perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];

end

function B = compute_B(X,A,invD,s)
    [d,n] = size(X);
    B = zeros(s);
    for i = 1:n,
        idxi = logical(A(:,i));
        s = sum(idxi);
        xi = X(:,i);
        Bi = X(:,idxi)-repmat(xi,1,s);
        invDi = invD(idxi,i);
        Bi = Bi.*repmat(invDi',d,1);
        B = B + Bi'*Bi;
    end
end

function Q = computeQ(Dist,A)
    s = sum(A(:,1));
    D = Dist(logical(A(:)));
    D = reshape(D,s,[]);
    Q = D./(repmat(sum(D),s,1));
end

function alpha  = update_alpha_smc(alpha_proximal,A,Q,alpha0,s,t,lambda)
    alpha = alpha0;
    n = size(alpha,2);
    
    P = t*eye(s);
    invAs = cell(1,n);
    lastprintlength = 0;
    lastprintlength = textprogressbar(-1,lastprintlength,'begin l1graph points: ');
    for i = 1:n,
        %idxi = logical(adjmat(:,i));
        %Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
        %Xt = X(:,idxi);
        idxi = logical(A(:,i));
        b1 = -t*alpha_proximal(:,i);
        b2 = lambda*Q(:,i);
        [outalpha,invAs] = admm_solve_alpha_smc(invAs,i,alpha(:,i),P,b1,b2);
        %alpha(idxi,i) = outalpha;
        alpha(:,i) = outalpha;
        lastprintlength = textprogressbar(0,lastprintlength,i,n);   
    end
    textprogressbar(1,lastprintlength,' Done\n');
end

function obj = compute_obj(B,A,alpha,invD,lambda,Q,gamma,s)
    obj = zeros(1,3);
    n = size(A,1);
    ralpha = reshape(alpha(logical(A(:))),s,[]);
    obj(1) = 0.5*trace(ralpha'*B*ralpha) + lambda*(Q(:)'*ralpha(:) );
    
    talpha_numerator = alpha.*invD;
    Rtalpha = repmat(sum(talpha_numerator),n,1);
    talpha = talpha_numerator ./ Rtalpha;
    
    obj(2) = gamma*norm(talpha*talpha-talpha,'fro')^2;
    obj(3) = obj(1) + obj(2); 
end



