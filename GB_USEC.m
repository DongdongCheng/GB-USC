function cluster_out = GB_USEC(fea, Ks)
cntTcutKmReps = 3; 
maxTcutKmIters = 100; 
K = 5; 
N = size(fea,1); %返回矩阵A所对应的行数
p = 5000;
if p>N
    p = N;
end
tic;
%%找到中心锚点
centerDist = getAnchor(fea, p);
[g,~]=size(centerDist);
disp(g);
toc;
%%获取前k个距离最小值
[knnIdx,knnDist] = knnsearch(centerDist,fea,'k',5);clear centerDist
knnMeanDiff = mean(knnDist(:)); 
Gsdx = exp(-(knnDist.^2)/(2*knnMeanDiff^2)); clear knnDist knnMeanDiff knnTmpIdx dataDist
Gsdx(Gsdx==0) = eps;
Gidx = repmat((1:N)',1,K);
B=sparse(Gidx(:),knnIdx(:),Gsdx(:),N,g); clear Gsdx Gidx knnIdx
labelscell = Tcut_for_bipartite_graph(B,Ks,maxTcutKmIters,cntTcutKmReps,N);
cluster_out = Sim_Manifold_binding(labelscell,Ks,0.1);

function last_center = getAnchor(fea, k)
[n, ~] = size(fea);
data1 = fea(randsample(n,k),:);
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

function gb_newcell1 = Split_again(hb_cell,radius)
gb_newcell1={};
e=1;
for j=1:length(hb_cell)
    if length(hb_cell{j}) <2
        gb_newcell1{e,1}=hb_cell{j};
        e=e+1;
    else
        [ball_1, ball_2] = spilt_ball(hb_cell{j});
        hb=cell2mat(hb_cell{j});
        if get_radius(hb)<=2*radius
            gb_newcell1{e,1}=hb_cell{j};
            e=e+1;
        else
            gb_newcell1{e,1} = ball_1;
            gb_newcell1{e+1,1} = ball_2;
            e = e+ 2;
        end
    end
end


function radius=get_radius(data)
[num,~]=size(data);
center=mean(data);
diffMat=repmat(center,num,1);
sqDiffMat=(diffMat-data).^2;
sqDistances =sum(sqDiffMat);
distances=sqrt(sqDistances);
radius=mean(distances);

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

 %%分裂球体
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

function labelcell = Tcut_for_bipartite_graph(B,Ks,maxKmIters,cntReps,N)
[Nx,Ny] = size(B);
if Ny < Ks
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

[e_vecs,eval] = eig(full(nWy)); clear nWy   
[~,idx] = sort(diag(eval),'descend');
labelcell= cell(1,10);
flag =0;
hlablel=Ks;
while 1
   flag=flag+1;
   Ncut_evec = D* e_vecs(:,idx(1:hlablel)); 
   evec = Dx * B * Ncut_evec; 
   evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );
   labels = kmeans(evec,Ks,'MaxIter',maxKmIters,'Replicates',cntReps);
   labelcell{1,flag}=labels;
   hlablel = hlablel +1;
   if flag ==10
       break
   end
end

function cluster_out = Sim_Manifold_binding(labelscell,Ks,g)
i = randperm(10,1);
randlabel = labelscell{i};
labelscell{i}=[];
labelmat = cell2mat(labelscell);clear labelscell
[~,n]=size(labelmat);
mapcell=cell(1,n+1);
for i= 1:n+1
    if i<=n
        maplabel= label_map(labelmat(:,i), randlabel);
        mapcell{i}=maplabel;
    else
        mapcell{i}=randlabel;
    end
end
labelmat = cell2mat(mapcell);clear mapcell maplabel randlabel
[~,t]=size(labelmat);
ave_NMIcell=cell(1,t);
for i=1:t
    sum_NMI=0;
    for j=1:t
        if i~=j
            result_NMI=computeNMI(labelmat(:,i),labelmat(:,j));
            sum_NMI = sum_NMI + result_NMI;
        end
    end
    ave_NMI = (1./( t-1)).*sum_NMI;
    ave_NMIcell{i}= ave_NMI;
end
ave_NMImat=cell2mat(ave_NMIcell);
ave_NMIsum = sum(ave_NMImat);
weigh =ave_NMImat./ave_NMIsum;
[~,r]=size(weigh);
[~,dataindex] = sort(weigh,2,'descend');%对行进行从高到低降序排序
e =round(r.*g);
a=r-e+1;
weigh(dataindex(a:end))=0;
[m,n]=size(labelmat);
cluster_out=zeros(m,1);
for ir = 1:size(labelmat,1)%行数
    Indic_mat=zeros(n,Ks);
    for ic = 1:size(labelmat,2)%列数
        index=labelmat(ir,ic);
        Indic_mat(ic,index)=1;     
    end
    get_allweigh=weigh*Indic_mat;
    [~,id]=max(get_allweigh);
    cluster_out(ir) = id;   
end





  
