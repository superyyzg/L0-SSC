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


addpath '/Users/ehsanelhamifar/Documents/MatlabCode/SMCE_v1.1_MotSeg';
addpath '/Users/ehsanelhamifar/Documents/MatlabCode/SemiDefiniteMethods/mve';

%%
%----- Clustering & Embedding of 2 Trefoil-Knots via SMCE

[Y,x,gtruth,tt] = manifoldGen('2trefoils'); lambda = 20; KMax = 20; dim = 2;
N = size(Y,2);
n = max(gtruth);
% plot the original data points
figure
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
axis off
%title('Trefoil-knots embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE algorithm
[Yc,Yj,clusters,missrate,W] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);
mscW = msc(W,clusters);
%K = 3; t = logspace(-1,3,50);
%[Yc,Yj,clusters,missrate,W] = LLE(Y,K,dim,n,gtruth);
%for p = 1:length(t)
%    [Yc,Yj,clusters,missrate(p),W] = LEM(Y,K,t(p),dim,n,gtruth);
%end

% [Yc,Yj,clusters,missrate(p),W] = LEM(Y,K,t(20),dim,n,gtruth);
%for i = 1:n
%    Yc{i} = Yc{i}';
%end

% plot the embedding(s)
for i = 1:n
    color{i} = jet(N);
    color{i} = color{i}(clusters==i,:);
    figure
    for j = 1:size(Yc{i}(end-1,:),2)
        plot(Yc{i}(end-1,j),Yc{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',9)
        hold on
    end
    axis('equal')
    axis off
    %ttl = strcat('Embedding of cluster ',num2str(i));
    %title(ttl,'fontsize',16)
end
if n > 1
    figure
    bar(mscW{1}(1:5))
    set(gcf,'Renderer','Painters')
    %axis([0.3 5.5 0 0.6])
    axis off
    figure
    bar(mscW{2}(1:5))
    set(gcf,'Renderer','Painters')
    %axis([0.5 5.5 0 0.8])
    axis off
else
    figure
    bar(mscW{1}(1:5))
    set(gcf,'Renderer','Painters')
    axis off
end

%%
%----- Embedding of 2D Sphere via SMCE

[Y,x,gtruth,tt] = manifoldGen('sphere'); 
%
lambda = 2; KMax = 100; dim = 2;
N = size(Y,2);
n = max(gtruth);
% plot the original data points
figure
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
axis off
%title('2D Sphere embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE algorithm
[Yc,Yj,clusters,missrate,W] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);
mscW = msc(W,clusters);

% plot the embedding(s)
figure
color{1} = jet(N);
for j = 1:size(Yc{1}(end-1,:),2)
    plot(Yc{1}(end-1,j),Yc{1}(end,j),'o','color',color{1}(j,:),'MarkerFaceColor',color{1}(j,:),'MarkerSize',6)
    hold on
end
axis('equal')
title('SMCE, \lambda = 2','FontName','Times New Roman','fontsize',56)
%title('Embedding of the sphere','fontsize',16)
axis off
set(gcf,'Renderer','Painters')


if n > 1
    figure
    bar(mscW{1}(1:10))
    set(gcf,'Renderer','Painters')
    %axis([0.3 5.5 0 0.6])
    axis off
    figure
    bar(mscW{2}(1:10))
    set(gcf,'Renderer','Painters')
    %axis([0.5 5.5 0 0.8])
    axis off
else
    figure
    bar(mscW{1}(1:10))
    set(gcf,'Renderer','Painters')
    axis off
end

%%
%----- Clustering & Embedding of Trefoil-knot and a Plane via SMCE

[Y,x,gtruth,tt] = manifoldGen('trefoil-plane-hole'); lambda = 40; KMax = 60; dim = 2;
N = size(Y,2);
n = max(gtruth);
% plot the original data points
figure
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
axis off
%title('Trefoil-knots embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

% verbose = true if want to see the sparse optimization information
verbose = true;

% run SMCE algorithm
%[Yc,Yj,clusters,missrate,W] = smce(Y,lambda,KMax,dim,n,gtruth,verbose);
%mscW = msc(W,clusters);
K = 7; t = logspace(-1,3,50);
%[Yc,Yj,clusters,missrate,W] = LLE(Y,K,dim,n,gtruth);
for p = 1:length(t)
    [Yc,Yj,clusters,missrate(p),W] = LEM(Y,K,t(p),dim,n,gtruth);
end
[pp,p] = min(missrate);
[Yc,Yj,clusters,missrate(p),W] = LEM(Y,K,t(5),dim,n,gtruth);

for i = 1:n
    Yc{i} = Yc{i}';
end

% plot the embedding(s)
for i = 1:n
    color{i} = jet(N);
    color{i} = color{i}(clusters==i,:);
    figure
    for j = 1:size(Yc{i}(end-1,:),2)
        plot(Yc{i}(end-1,j),Yc{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',9)
        hold on
    end
    axis('equal')
    axis off
    %ttl = strcat('Embedding of cluster ',num2str(i));
    %title(ttl,'fontsize',16)
end
if n > 1
    figure
    bar(mscW{1}(1:5))
    set(gcf,'Renderer','Painters')
    %axis([0.3 5.5 0 0.6])
    axis off
    figure
    bar(mscW{2}(1:5))
    set(gcf,'Renderer','Painters')
    %axis([0.5 5.5 0 0.8])
    axis off
else
    figure
    bar(mscW{1}(1:5))
    set(gcf,'Renderer','Painters')
    axis off
end
