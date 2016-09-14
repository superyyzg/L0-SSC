function [alpha,invAs] = update_alpha1(X,invAs,alpha0,L0,Y1,beta,lambda,adjmat,dic_size) 

n = size(X,2);
alpha = alpha0;

K = (beta*L0);
n = size(K,1);
K_diag = diag(K);
K_diagr = repmat(K_diag,1,n);
K_diagc = repmat(K_diag',n,1);
cK = 0.5* (K_diagr + K_diagc - K - K');

lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,'begin l1graph points: ');
for i = 1:n,
    idxi = logical(adjmat(:,i));
    %Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
    
    H1  = (zeros(n)); H1(i,:) = 1;
    H2  = (zeros(n)); H2(i,i) = 1;
    
    Xt = X(:,:);
    XtX = Xt'*Xt;
    
    P = beta/2*(0.5*H1 + 3/4*eye(n) + 1/4*ones(n) - 1/2*H2 - H1);
    
    
    alpha_colsum = (sum(alpha))';
    alpha_rowsum = sum(alpha,2);
    Q1 = alpha_colsum + alpha_rowsum(i) - alpha(i,i); Q1(i) = alpha_rowsum(i) - alpha(i,i);
    Q2 = 2*(alpha_rowsum - alpha(:,i)); 
    Q3 = 2*alpha(i,:)'; Q3(i) = 0;
    Q4 = diag(alpha); Q4(i) = alpha_rowsum(i) - alpha(i,i);
    b = beta/2*(1/2*Q1 + 1/4*Q2 + 1/2*Q3 - Q4)  - cK(:,i);
    
    %P = P(idxi,idxi); b = b(idxi);
    P = P + XtX; P = P*2;
    b = b -2*Xt'*X(:,i) + ones(n,1)*lambda;
    
    
    [outalpha,invAs] = admm_solve_qp(invAs,i,alpha(:,i),P,b);
    
    alpha(:,i) = outalpha;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);   
end
textprogressbar(1,lastprintlength,' Done\n');