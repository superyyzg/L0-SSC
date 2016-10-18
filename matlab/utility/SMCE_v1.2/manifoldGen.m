%--------------------------------------------------------------------------
% This function generates manifols 'sphere' or '2trefoils'
% D = dimension of the ambient space
% sigma = variance of the noise added to the data
% N = number of points in each manifold
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Yn,Y,gtruth,x] = manifoldGen(manifoldType)


rand('state',10000);
randn('state',10000);

D = 100; % ambient space dimension
sigma = 0.001; % noise variance

if (strcmp(manifoldType,'2trefoils'))
    N = 100;
    gtruth = [1*ones(1,N) 2*ones(1,N)];
elseif (strcmp(manifoldType,'sphere'))
    N = 1000;
    gtruth = ones(1,N);
elseif (strcmp(manifoldType,'trefoil-plane-hole'))
    N = [200 400];
    %gtruth = [1*ones(1,N(1)) 2*ones(1,N(2))];
else
    error('Unknown Mnifold Type!')
end

if (strcmp(manifoldType,'2trefoils'))
        % Generate two trefoil-knots
        d = 3; Par = 3.8;
        t = linspace(0,2*pi,N+1);
        t(end) = [];
        Yg{1}(1,:) = (2+cos(3*t)).*cos(2*t);
        Yg{1}(2,:) = (2+cos(3*t)).*sin(2*t);
        Yg{1}(3,:) = sin(3*t);
        Yg{2}(1,:) = (2+cos(3*t)).*cos(2*t) + Par;
        Yg{2}(2,:) = (2+cos(3*t)).*sin(2*t);
        Yg{2}(3,:) = sin(3*t);
        Y = [Yg{1} Yg{2}];
        U = orth(randn(D,d));
        Yn = U * Y;
        Yn = Yn + sigma * randn(size(Yn));
        x = [cos(t) cos(t);sin(t) sin(t)];
        
elseif (strcmp(manifoldType,'sphere'))
        % Generate a random sphere
        d = 3;
        r = 2 * ( sort(rand(1,N)).^.99);
        theta = linspace(0,2*pi,N+1);
        theta(end) = [];
        p = randperm(N);
        P = zeros(N,N);
        for i = 1:N
            P(p(i),i) = 1;
        end
        theta = theta * P;
        xx = r .* cos(theta);
        yy = r .* sin(theta);        
        Y = [2*xx./(1+xx.^2+yy.^2);2*yy./(1+xx.^2+yy.^2);(-1+xx.^2+yy.^2)./(1+xx.^2+yy.^2)];
        U = orth(randn(D,d));
        Yn = U * Y;
        Yn = Yn + sigma * randn(size(Yn));
        x = [xx;yy];
elseif (strcmp(manifoldType,'trefoil-plane-hole'))
    Par = 1;
    % Generate a trefoil and a plane with a hole in it
    t = sort( 2*pi*rand(1,N(1)+1) ); t(end) = [];
    Yg{1}(1,:) = (2+cos(3*t)).*cos(2*t);
    Yg{1}(2,:) = (2+cos(3*t)).*sin(2*t);
    Yg{1}(3,:) = sin(3*t);
    [B,~,~] = svd([1;.4;.8]);
    L = sqrt(N(2));
    a1 = linspace(-3,3,L);
    a2 = linspace(-3,3,L);
    for i = 1:L
        for j = 1:L
            Yg{2}(:,(i-1)*L+j) = [a1(i);a2(j);1];
        end
    end
    Yg{2}(:,sum(Yg{2}.^2,1) <= 2)=[];
    [~,ihs] = sort(Yg{2}(1,:),'ascend');
    for p = 1:size(Yg{2},1)
        Yg{2}(p,:) = Yg{2}(p,ihs);
    end
    N = [size(Yg{1},2) size(Yg{2},2)];
    Yg{2}(1:2,:) = Yg{2}(1:2,:) + .1 * randn(2,N(2));
    Yg{2} = B * Yg{2};
    Yg{2}(1,:) = Yg{2}(1,:) + 4 + Par;
    Yg{2}(3,:) = Yg{2}(3,:) - 1;
    Y = [Yg{1} Yg{2}];
    Yn = Y + 0.01 * randn(3,size(Y,2));
    x = [cos(t) Yg{2}(1,:);sin(t) Yg{2}(2,:)];
    gtruth = [1*ones(1,N(1)) 2*ones(1,N(2))];
end



