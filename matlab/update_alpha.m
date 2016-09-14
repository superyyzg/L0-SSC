function [alpha,invAs] = update_alpha(X,iter,invAs,alpha0,hatalpha0,Y2,beta,lambda,adjmat,dic_size) 

n = size(X,2);
alpha = alpha0;

lastprintlength = 0;
lastprintlength = textprogressbar(-1,lastprintlength,'begin l1graph points: ');
for i = 1:n,
    idxi = logical(adjmat(:,i));
    %Xflag = ones(1,n); Xflag(i) = 0; Xflag = logical(Xflag);
    Xt = X(:,idxi);
    XtX = Xt'*Xt;
    P = 2*XtX + beta*eye(size(XtX,1));
    b1 = -2*Xt'*X(:,i) - Y2(idxi,i) - beta*hatalpha0(idxi,i);
    b2 = ones(dic_size,1)*lambda;
    [outalpha,invAs] = admm_solve_alpha(iter,invAs,i,alpha(idxi,i),P,b1,b2);
    alpha(idxi,i) = outalpha;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);   
end
textprogressbar(1,lastprintlength,' Done\n');