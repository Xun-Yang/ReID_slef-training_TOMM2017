%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Demo code for the following paper (VIPeR and CUHK01):
%%%
%%% Xun Yang, Meng Wang, Richang Hong, Qi Tian, and Yong Rui. 
%%% "Enhancing Person Re-identification in a Self-trained Subspace". 
%%% ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2017.(https://arxiv.org/abs/1704.06020)
%%%
%%% Note that this paper was just accepted by TOMM. It may take several months to appear on TOMM.
%%%
%%% Pls feel free to contact us for any questions (hfutyangxun@gmail.com).
%%%
%%% 
%%% Pls also cite the following paper if you use this code:
%%%
%%%    Tetsu Matsukawa, Takahiro Okabe, Einoshin Suzuki, Yoichi Sato. “Hierarchical Gaussian Descriptor for Person Re-Identification”.     
%%%    in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR2016), pp.1363--1372, 2016 
%%% 
%%% research purpose only.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
addpath('./utils');

%% configuraiton of datasets. 
  sys.database =1;      dataname='VIPeR';
%sys.database =2;    dataname='CUHK01M1';
 %sys.database =3;    dataname='CUHK01M2';
   sys.setnum = 10;
   
   option.theta= 0.01;
   option.labelRatio= 1/3;
   option.nIter=10;
   option.k= 2;
   option.eta= 1;
   option.kernelFunc={'Gaussian'};

%% algoname setting
   algoname= 'MKSSL';
%%
config;
load_features_all; % load all features.

CMCs = zeros( sys.setnum, numperson_garalley );
trainnum = numperson_train;
numRanks=100;

   for set = 1:sys.setnum
    fprintf('----------------------------------------------------------------------------------------------------\n');
    fprintf('set = %d \n', set);
    fprintf('----------------------------------------------------------------------------------------------------\n');
    
    %% Training data
     tot = 1;
%% The first feature
    extract_feature_cell_from_all;  % load training data
    apply_normalization; % feature normalization
    conc_feature_cell; % feature concatenation
    traininds=traininds_set{set};
    camIDs = traincamIDs_set{set};
  
    probTrain = normc(feature(camIDs == 1, :)')';
    galTrain =   normc(feature(camIDs == 2, :)')';
    
    labels = trainlabels_set{set};
    probLabels = labels(camIDs == 1);
    galLabels = labels(camIDs == 2);
    
        [model, kernelparam, trainFea]= MKSSL(galTrain, probTrain, galLabels, probLabels, option);
        P=model.P;
        M=eye(size(P,2)); 

    clear camIDs  galLabels probLabels galTrain probTrain 
    
    %% Test data
    tot = 2;
    extract_feature_cell_from_all; % load test data
    apply_normalization; % feature normalization
    conc_feature_cell; % feature concatenation
    
    testinds=testinds_set{set};
    camIDs = testcamIDs_set{set};

    probTest = normc(feature(camIDs == 1, :)')';
    galTest =   normc(feature(camIDs == 2, :)')';
     
         
    labels = testlabels_set{set};
    labelsPr = labels(camIDs == 1);
    labelsGa = labels(camIDs == 2);
   
    galX=   KernelTest(trainFea, galTest, kernelparam,  option.kernelFunc); 
    probX= KernelTest(trainFea, probTest, kernelparam, option.kernelFunc);
    
        
     clear  galTest   probTest  
   
    if sys.database ~= 3
        % single shot matching
      scores = MahDist(M, galX * P, probX * P)';
       CMC = zeros( numel(labelsGa), 1);
        
          for p=1:numel(labelsPr)
              score = scores(p, :);
              [sortscore, ind] = sort(score, 'ascend');
              correctind = find( labelsGa(ind) == labelsPr(p));
              CMC(correctind:end) = CMC(correctind:end) + 1;
          end
         
        CMC = 100.*CMC/numel(labelsPr);
        CMCs(set, :) = CMC;
        
    else
        % multishot matching for CUHK01 
        probX1 = probX(1:2:size(probX, 1), :);
        probX2 = probX(2:2:size(probX, 1), :);
        galX1 =   galX(1:2:size(galX, 1), :);
        galX2 =   galX(2:2:size(galX, 1), :);
        
        labelsPr1 = labelsPr(1:2:size(probX, 1), 1);
        labelsPr2 = labelsPr(2:2:size(probX, 1), 1);
        labelsGa1 = labelsGa(1:2:size(galX, 1), 1);
        labelsGa2 = labelsGa(2:2:size(galX, 1), 1);
        
        scores1 = MahDist(M, galX1 * P, probX1 * P)';
        scores2 = MahDist(M, galX2 * P, probX1 * P)';
        scores3 = MahDist(M, galX1 * P, probX2 * P)';
        scores4 = MahDist(M, galX2 * P, probX2 * P)';

        scores = scores1 + scores2 + scores3 + scores4;
        
        CMC = zeros( numel(labelsGa1), 1);
        for p=1:numel(labelsPr1)
            score = scores(p, :);
            [sortscore, ind] = sort(score, 'ascend');
            
            correctind = find( labelsGa1(ind) == labelsPr1(p));
            CMC(correctind:end) = CMC(correctind:end) + 1;
        end
        CMC = 100.*CMC/numel(labelsPr1);
        CMCs(set, :) = CMC;
    end
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', CMC([1,5,10,15,20]));
    
   end
        clear camIDs probX galX galLabels probLabels options model
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf('  Mean Result \n');
fprintf('----------------------------------------------------------------------------------------------------\n');
% clear set;
clear name meanCms
meanCms = mean( squeeze(CMCs(1:sys.setnum , :)), 1);
fprintf(' Rank1, Rank5, Rank10, Rank20 \n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%% \n', meanCms([1,5,10,20]) );

figure;
hold on
plot(1 : numRanks/2, meanCms(1 : numRanks/2),'-b','LineWidth',2);
Value=num2str(meanCms(1));
name1=sprintf([Value(1:min(length(Value),5)) '%%-' algoname]);
legend(name1,'Location','SouthEast');
grid on
box on

 title(['CMC Curve on ' databasename ' (ratio=1/' num2str(1/option.labelRatio) ')'], 'fontsize', 12);
 xlabel('Rank score', 'fontsize', 12);
 ylabel('Recognition rate', 'fontsize', 12);
hold off


