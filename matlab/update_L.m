function [L] = update_L(LW,Y1,beta,k,eps0,thr)

addpath('.\utility\PROPACK');

L1 = (LW - Y1/beta);

n = size(L1,1);

[U1,S1,V1] = svd(L1);

%[U,S,V] = lansvd(L1,n,'S');

dS1 = diag(S1); dS1k = dS1(n-k+1:n);

if sum(dS1k) <= eps0 , 
    L = L1;
else
    dS = zeros(k,1);
    max_L_iter = 50;
    
    for iter = 1:max_L_iter,
        for i = 1:k,
            if eps0 - sum(dS)+dS(i) >= dS1k(i),
                dS(i) = dS1k(i);
            else
                dS(i) = eps0 - sum(dS)+dS(i);
            end
        end
        if iter > 1,
            err = errorCoef(dS,dS0);
        else
            err = 10*thr;
        end
        if (err < thr),
            break;
        end
        
        dS0 = dS;
    end
    
    dS = [dS1(1:n-k);dS];
    L = U1*diag(dS)*V1';
end






