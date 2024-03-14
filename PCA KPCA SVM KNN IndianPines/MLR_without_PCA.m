%% HyperSpectral Classification
clear memory
close all;
clear all;
clc
% spectral_data=load('PaviaU.mat');%Hyperspectral
% gt_data=load(p'PaviaU_gt.mat');%Groundtruth
% 
%spectral_data=load('Salinas.mat');
%gt_data=load('Salinas_gt.mat');

spectral_data=load('Indian_pines.mat');
gt_data=load('Indian_pines_gt.mat');



%%
%% Spectral Classi
addpath(genpath(pwd));
%%
class_dict={'Alfalfa','Corn-notill',...
    'Corn-mintill','Corn','Grass-pasture',...
    'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed',...
    'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean',...
    'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'};
dataInfo=struct('data_name','Salinas','num_class',16,'class_dict',{class_dict},...
       'spectral_data',spectral_data.indian_pines,'gt_data',gt_data.indian_pines_gt);
   
%% 
%figure;subplot(1,2,1);imagesc(dataInfo.gt_data);

dataInfo.splitInfo=SplitData(dataInfo.gt_data,'fix_num',100,true);

train_feat=VectorIndexing3D(dataInfo.spectral_data,dataInfo.splitInfo.train_idx);
test_feat=VectorIndexing3D(dataInfo.spectral_data,dataInfo.splitInfo.test_idx);
train_feat=train_feat/1000;
test_feat=test_feat/1000;
train_label=dataInfo.splitInfo.train_label;
test_label=dataInfo.splitInfo.test_label;
%%
%%
disp('Perform classification.');
%% MLR Sparse Multinomial Logistic Regression 
% Implements a block Gauss Seidel  algorithm for fast solution of 
%  the SMLR

[w,val_loss] = MLRTrainAL(train_feat',train_label', 0.01,0.01,500);
[pred_label,pred_prob]=MLREval(test_feat,16, w);


%%
[clsStat,mat_conf]=GetAccuracy(test_label,pred_label);
disp(['Overall Accuracy:',num2str(clsStat.OA),', Kappa Coeffcient:', num2str(clsStat.Kappa)]);
color_map=GetColorMap(64);
mat_pred_label=zeros(size(dataInfo.gt_data));
mat_pred_label(dataInfo.splitInfo.test_idx)=pred_label;
class_map=GetClassMap(mat_pred_label,color_map);
class_map_gt=GetClassMap(dataInfo.gt_data,color_map);


figure('Name','MLR Without PCA');
subplot(1,3,1);imagesc(dataInfo.gt_data);  title('Original gt');
subplot(1,3,2);imagesc(class_map_gt);title('class map gt');
subplot(1,3,3);imagesc(class_map);title('Result');

