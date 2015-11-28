function Mspectral()
%parameters dataset1
%{
clusterN = 2;
featureN = 24;
dataN = 1000;
edge = 9;
filename = 'german.txt';
%}

%parameters dataset2

clusterN = 10;
featureN = 784;
dataN = 10000;
edge = 9;
filename = 'mnist.txt';


data = zeros(dataN,featureN);
datatag = zeros(dataN,2);
Knear = zeros(dataN,edge);
I = [];
J = [];
V = [];

%file input
t0 = cputime;
fprintf('%s','file input...');
fid = fopen(filename,'r');
for i = 1 : dataN
    data(i,:) = fscanf(fid,'%g,',featureN);
    datatag(i,1) = fscanf(fid,'%d',1);
end
fclose(fid);
fprintf('\t%s -usage time:%fs\n','[done]',cputime-t0);


t0 = cputime;
fprintf('%s','calculate k-near...');
for i = 1:dataN/1000
    i1 = 1000*(i-1)+1;
    i2 = 1000*i;
    x = data(i1:i2,:);
    y = data;
    dist = sqrt(-bsxfun(@minus,bsxfun(@minus,2*x*y',sum(x'.^2,1)'),sum(y'.^2,1)));
    [~, index] = sort(dist,2);
    Knear(i1:i2,:) = index(1:i2-i1+1,1:edge);
end    
fprintf('\t%s -usage time:%fs\n','[done]',cputime-t0);


t0 = cputime;
fprintf('%s','construct graph...');
for i = 1: dataN
    for j = 1: dataN
        have = 0;
        for k = 1 : edge
            if (Knear(i,k) == j || Knear(j,k) == i)
                have = 1;
                break;
            end
        end
        if (have == 1)
            I = [I,i];
            J = [J,j];
            V = [V,1.0];
        end
    end
end
W = sparse(I,J,V,dataN,dataN);
fprintf('\t%s -usage time:%fs\n','[done]',cputime-t0);

t0 = cputime;
fprintf('%s','calculate tag...');
dsum = sum(W, 2);
D = sparse(1:size(W,1),1:size(W,2), dsum);
L = D - W;
[Vector,Value] = eigs(L, clusterN+1, 'SA');
Tag = kmeans(Vector(:,2:clusterN+1), clusterN);
datatag(:,2) = Tag;

fprintf('\t%s -usage time:%fs\n','[done]',cputime-t0);

t0 = cputime;
fprintf('%s','calculate validation...');

G = zeros(clusterN,1);
m = zeros(clusterN,clusterN);
for i = 1:dataN
    m(datatag(i,1)+1,datatag(i,2)) = m(datatag(i,1)+1,datatag(i,2)) + 1;
end
N = sum(m,2);
M = sum(m,1);
P = max(m);

a = sum(P);
b = sum(M);
purity = a/b;

for j = 1:clusterN
    for i = 1: clusterN
        if (m(i,j) ~= 0)
            G(j) = G(j) + (m(i,j)/M(j)) * (m(i,j)/M(j));
        end
    end
    G(j) = 1 - G(j);
end
a = 0;
b = 0;
for j = 1: clusterN
    a = a + G(j) * M(j);
    b = b + M(j);
end
gini = a/b;
fprintf('\t%s -usage time:%fs\n','[done]',cputime-t0);
fprintf('purity is : %f , gini is : %f \n',purity,gini);

end



