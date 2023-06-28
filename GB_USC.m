function [labels,g] = GB_USC(fea, Ks)
cntTcutKmReps = 3; 
maxTcutKmIters = 100; 
K = 5; % The number of nearest neighbors.
p = 5000; 
N = size(fea,1);
if p>N
    p = N;
end
%%找到中心锚点
centerDist = getAnchor(fea, p);
[g,~]=size(centerDist);
disp(g);

[knnIdx,knnDist] = knnsearch(centerDist,fea,'k',5);
knnMeanDiff = mean(knnDist(:)); 
Gsdx = exp(-(knnDist.^2)/(2*knnMeanDiff^2)); clear knnDist knnMeanDiff knnTmpIdx dataDist
Gsdx(Gsdx==0) = eps;
Gidx = repmat((1:N)',1,K);
B=sparse(Gidx(:),knnIdx(:),Gsdx(:),N,g); clear Gsdx Gidx knnIdx

labels = zeros(N, numel(Ks));
for iK = 1:numel(Ks)
    [labels(:,iK),evec] = Tcut_for_bipartite_graph(B,Ks(iK),maxTcutKmIters,cntTcutKmReps);
end

%%%获得锚点
function last_center = getAnchor(fea, k)
[n, ~] = size(fea);
data1 = fea(randsample(n,k),:);%从数据集中随机取2000个数据
data_cell = {data1};
while 1
    ball_number_old = length(data_cell);
    data_cell = division(data_cell);
    ball_number_new = length(data_cell);
    if ball_number_new == ball_number_old
       break
    end
end
m=length(data_cell);
last_center=cell(m,1);
last_radius=cell(m,1);
for h=1:m
    data = cell2mat(data_cell{h});
    last_radius{h,1}=get_radius(data);
    if size(data,1)==1
        last_center{h,1}=data;
    else  
        last_center{h,1}=mean(data);
    end
end
last_center=cell2mat(last_center);
last_radius=cell2mat(last_radius);

%%%再一次分裂
function gb_new = Split_again(hb_cell,radius)
gb_new={};
e=1;
for j=1:length(hb_cell)
    if length(hb_cell{j}) <2
        gb_new{e,1}=hb_cell{j};
        e=e+1;
    else
        [ball_1, ball_2] = spilt_ball(hb_cell{j});
        hb=cell2mat(hb_cell{j});
        if get_radius(hb)<=2*radius
            gb_new{e,1}=hb_cell{j};
            e=e+1;
        else
            gb_new{e,1} = ball_1;
            gb_new{e+1,1} = ball_2;
            e = e+ 2;
        end
    end
end


%%获得半径
function radius=get_radius(data)
[num,~]=size(data);
center=mean(data);
diffMat=repmat(center,num,1);
sqDiffMat=(diffMat-data).^2;
sqDistances =sum(sqDiffMat);
distances=sqrt(sqDistances);
radius=mean(distances);

%%分裂球
function gb_newcell = division(hb_cell)
gb_newcell={};
i=1;
 for j =1:length(hb_cell)
     [m,~]=size(hb_cell{j});
     if m>7
            [ball_1, ball_2] = spilt_ball(hb_cell{j});
            gb_newcell{i,1} = ball_1;
            gb_newcell{i+1,1} = ball_2;
            i = i+ 2;
     else
            gb_newcell{i,1} = hb_cell{j};
            i = i+1;  
     end
 end

%%%定义分裂球
function [ball_1, ball_2] = spilt_ball(data)
if iscell(data)
    data = cell2mat(data);
end
[n, ~] = size(data);
ball_1 = {};
ball_2 = {};
D=pdist(data);
D=squareform(D);
[r,c] = find(D == max(max(D))); 
r1 = r(2);
c1 = c(2);
i = 1;
k = 1;
for j = 1:n
    if D(j,r1) < D(j,c1)
        ball_1{i,1} = data(j,:);
        i = i+1;
    else
        ball_2{k,1} = data(j,:);
        k = k+1;
    end
end

function [labels,evec] = Tcut_for_bipartite_graph(B,Nseg,maxKmIters,cntReps)
if nargin < 4
    cntReps = 3;
end
if nargin < 3
    maxKmIters = 100;
end

[Nx,Ny] = size(B);
if Ny < Nseg
    error('Need more columns!');
end

dx = sum(B,2);
dx(dx==0) = 1e-10;
Dx = sparse(1:Nx,1:Nx,1./dx); clear dx
Wy = B'*Dx*B;

d = sum(Wy,2);
D = sparse(1:Ny,1:Ny,1./sqrt(d)); clear d
nWy = D*Wy*D; clear Wy
nWy = (nWy+nWy')/2;


[evec,eval] = eig(full(nWy)); clear nWy   
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg)); clear D

evec = Dx * B * Ncut_evec; clear B Dx Ncut_evec

evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );

% k-means
labels = kmeans(evec,Nseg,'MaxIter',maxKmIters,'Replicates',cntReps);


