%--------------------------------------------------------------------------
% This is a demo to run SMCE
% lambda = regularization paramter for the Lasso-type optimization program
% KMax = maximum neighborhood size to select the sparse neighbors from
% dim = dimension of the low-dimensional embedding
% n = number of clusters/manifolds
% Yg = has n cells where each cell contains the dim-dimensional embedding
% of the data in each cluster
% gtruth = ground-truth memberships of points to manifolds (if available)
% missrate = clustering error when n > 1
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

clc, clear all, warning off


%----- Clustering & Embedding of 2 Trefoil-Knots via SMCE

[Y,x,gtruth] = manifoldGen('2trefoils'); lambda = 10; KMax = 50; dim = 2;
N = size(Y,2);
n = max(gtruth);
% plot the original data points
figure(1)
subplot(221)
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
title('Trefoil-knots embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE algorithm
[Yc,Yj,clusters,missrate] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);


% plot the embedding(s)
f = 2;
for i = 1:n
    color{i} = jet(N);
    color{i} = color{i}(clusters==i,:);
    f = f + 1;
    eval(['subplot(22' num2str(f) ')'])
    for j = 1:size(Yc{i}(end-1,:),2)
        plot(Yc{i}(end-1,j),Yc{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',9)
        hold on
    end
    axis('equal')
    ttl = strcat('Embedding of cluster ',num2str(i));
    title(ttl,'fontsize',16)
end


%----- Embedding of 2D Sphere via SMCE

[Y,x,gtruth] = manifoldGen('sphere'); lambda = 10; KMax = 50; dim = 2;
N = size(Y,2);
n = max(gtruth);
% plot the original data points
figure(2)
subplot(211)
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
title('2D Sphere embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE algorithm
[Yc,Yj,clusters,missrate,W] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);

% plot the embedding(s)
subplot(212)
color{1} = jet(N);
for j = 1:size(Yc{1}(end-1,:),2)
    plot(Yc{1}(end-1,j),Yc{1}(end,j),'o','color',color{1}(j,:),'MarkerFaceColor',color{1}(j,:),'MarkerSize',9)
    hold on
end
axis('equal')
title('Embedding of the sphere','fontsize',16)

