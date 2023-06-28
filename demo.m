function demo()
%% Run the GB_USPEC algorithm multiple times and show its average performance.


clear
close all;

%% Load the data.
%dataName = 'PenDigits';     
%dataName = 'USPS';
%dataName = 'MNIST';
dataName ='Iris';

%dataName ='t4';         %%%人工数据集
%dataName ='E6_nonoise'; 
%dataName ='D3';
%dataName='aggregation';
%dataName = 'TB_1048576';
%dataName = 'SF_1048576';
%dataName = 'CC_1048576';
%dataName = 'CG_1048576';
%dataName = 'Flower_1048576';

% Load the dataset.
gt= [];
fea = [];
load(['data_',dataName,'.mat'],'fea','gt'); 
%load('Salinas.mat') 
[N, d] = size(fea);

%% Set up
k = numel(unique(gt)); % The number of clusters
cntTimes = 20;

%% Run 
FmeasureScores = zeros(cntTimes,1);
accScores = zeros(cntTimes,1);
runtime =  zeros(cntTimes,1);
disp('.');
disp(['N = ',num2str(N)]);
disp('.');
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    disp('.');
    disp('Performing GB-USC ...');
    disp('.');
    
    tic;
    Label = GB_USC(fea, k);
    %Label = GB_USEC(fea, k);
    toc;
    runtime(runIdx) = toc;
    disp('.');
    disp('--------------------------------------------------------------');
    
    [new_label] = label_map( Label, gt);
    [fmeasure,~]=Fmeasure(gt,new_label);
    FmeasureScores(runIdx)=fmeasure;
    accScores(runIdx) = accuracy(gt,new_label);
  
    disp(['The ACC score at Run ',num2str(runIdx), ': ',num2str(accScores(runIdx))]);
    disp(['The Fmeasure score at Run ',num2str(runIdx), ': ',num2str(FmeasureScores(runIdx))]);
    disp(['The time score at Run ',num2str(runIdx), ': ',num2str(runtime(runIdx))]);
    disp('--------------------------------------------------------------');
end

disp('**************************************************************');
disp(['  ** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Sample size: N = ', num2str(N)]);
disp(['Dimension:   d = ', num2str(d)]);
disp('--------------------------------------------------------------');
disp(['Average ACC score: ',num2str(mean(accScores))]);
disp(['Average Fmeasure score: ',num2str(mean(FmeasureScores))]);
disp(['Average RUNTIME: ',num2str(mean(runtime))]);
disp('--------------------------------------------------------------');
disp('**************************************************************');