function F=fmeasure(u,v)

%INPUTS
% u = the labeling as predicted by a clustering algorithm
% v = the true labeling
%
%OUTPUTS
% adjrand = the adjusted Rand index

n=length(u);
ku=max(u);
kv=max(v);
m=zeros(ku,kv);
for i=1:n
    m(u(i),v(i))=m(u(i),v(i))+1;
end
mu=sum(m,2);
mv=sum(m,1);

nTParis = 0;
for i=1:kv
    if mv(i)>1
        nTParis = nTParis + nchoosek(mv(i),2);
    end
end

nPPairs = 0;
for i=1:ku
    if mu(i)>1
        nPPairs = nPPairs + nchoosek(mu(i),2);
    end
end

nPTPairs = 0;
for i = 1:n,
    for j = i+1:n,
        if ((u(i) == u(j)) && (v(i) == v(j))),
            nPTPairs = nPTPairs + 1;
        end
    end
end

% for i=1:ku
%     for j=1:kv
%         if m(i,j)>1
%             nPTPairs=nPTPairs+nchoosek(m(i,j),2);
%         end
%     end
% end

Recall = nPTPairs/nTParis;
Precision = nPTPairs/nPPairs;

if (Recall == 0) || (Precision == 0),
    F = 0;
else
    F = 2*Recall*Precision/(Recall+Precision);
end

