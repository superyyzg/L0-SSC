function [hatalpha] = update_hatalpha(invAs,alpha,L0,Y1,Y2,beta,adjmat,thr)

mat_alpha = alpha - Y2/beta;

K = (beta*L0 + Y1);
n = size(K,1);
K_diag = diag(K);
K_diagr = repmat(K_diag,1,n);
K_diagc = repmat(K_diag',n,1);
cK = 0.5* (K_diagr + K_diagc - K - K');

max_hatalpha_iter = 50;

hatalpha = alpha;

for iter = 1:max_hatalpha_iter,
    
 
lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,['begin update_hatalpha iter ', num2str(iter), ':']);
for i = 1:n,
    
    
    H1  = (zeros(n)); H1(i,:) = 1;
    H2  = (zeros(n)); H2(i,i) = 1;
    H3  = (zeros(n)); H3(i,:) = 1; H3(i,i) = 0;
    P = beta/2*(0.5*H1 + 3/4*eye(n) + 1/4*ones(n) - 1/2*H2 - H1 + eye(n));
    P = P*2;
    
    hatalpha_colsum = (sum(hatalpha))';
    hatalpha_rowsum = sum(hatalpha,2);
    Q1 = hatalpha_colsum + hatalpha_rowsum(i) - hatalpha(i,i); Q1(i) = hatalpha_rowsum(i) - hatalpha(i,i);
    Q2 = 2*(hatalpha_rowsum - hatalpha(:,i)); 
    Q3 = 2*hatalpha(i,:)'; Q3(i) = 0;
    Q4 = diag(hatalpha); Q4(i) = hatalpha_rowsum(i) - hatalpha(i,i);
    b = beta/2*(1/2*Q1 + 1/4*Q2 + 1/2*Q3 - Q4) - beta*abs(mat_alpha(:,i)) - cK(:,i);
    
    idxi = logical(adjmat(:,i));
    P = P(idxi,idxi); b = b(idxi);
    [outhatalpha,invAs] = admm_solve_qp(invAs,i,hatalpha(idxi,i),P,b);
    hatalpha(idxi,i) = outhatalpha;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);   
end
textprogressbar(1,lastprintlength,'update_hatalpha iter Done\n');

if iter > 1,
    err = errorCoef(hatalpha,hatalpha0);
else
    err = 10*thr;
end

if (err < thr),
    break;
end

hatalpha0 = hatalpha;

end

hatalpha = sign(mat_alpha).*hatalpha;


function obj = compute_obj(alpha,hatalpha,beta)

hatalpha = sparse(hatalpha); 






