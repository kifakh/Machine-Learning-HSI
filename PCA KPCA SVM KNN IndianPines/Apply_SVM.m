%% HIC 6 2021
%%
clear; clc; close all;

%% Params
% User Params
%% Ratio(TrainData/Groundtruth)=0.01, 0.1, 0.3
perc_rate = 0.1;   

verbose = true;
% kernel_type = 'RBF';
%kernel_type = 'polynomial';
 kernel_type = 'linear';
 
%% Load Images
hyperdata = load('indian_pines.mat');
% hyperdata = hyperdata.salinas;
hyperdata = hyperdata.indian_pines;
% hyperdata = hyperdata.paviaU; 

hyperdata_gt = load('indian_pines_gt.mat');
% hyperdata_gt = hyperdata_gt.salinas_gt; 
hyperdata_gt = hyperdata_gt.indian_pines_gt;
% hyperdata_gt = hyperdata_gt.paviaU_gt;
%% Normalization
%hyperdata = ( hyperdata - min(min(min(hyperdata))) ) /  ( max(max(max(hyperdata))) - min(min(min(hyperdata))) );
[h,w,spec_size] = size(hyperdata);
% data2vector
hypervector = reshape(hyperdata , [h*w,spec_size]);

%% 	Variables

Mask = zeros(size(hyperdata_gt));       
NumOfClass = max(hyperdata_gt(:));      
NumOfClassElements = zeros(1,NumOfClass);       
%% Calculate each class's elements number

for i = 1 : 1 : NumOfClass
    NumOfClassElements(i) = double(sum(sum((hyperdata_gt == i)))); 
end

%% Perc_Rate data will select for training vector 
for i = 1 : 1 : NumOfClass
    perc_5 = floor(NumOfClassElements(i) * perc_rate);
    [row,col] = find(hyperdata_gt == i);
    
    while(sum(sum(Mask == i)) ~= perc_5) 
        x = floor((rand() * (NumOfClassElements(i) - 1)) + 1);
        Mask(row(x),col(x)) = hyperdata_gt(row(x),col(x));
    end
end

%%
spectral_data=load('Indian_pines.mat');
gt_data=load('Indian_pines_gt.mat');

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
%% Class's Nums

ClassSayisi = 0;
for i = 1 : 1 : NumOfClass
   ClassSayisi = ClassSayisi + sum(sum(Mask == i)); 
end

%% Created Train and Label Vector %%etiket ve eðitimmm miii

[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
trainingVector = zeros(ClassSayisi,spec_size);
trainingVectorLabel = zeros(ClassSayisi,1);

%% Training Vector & Training Vector Label 

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train
tic;
classes = unique(trainingVectorLabel);
num_classes = numel(classes);
svms = cell(num_classes,1);

for k=1:NumOfClass
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    class_k_label = trainingVectorLabel == classes(k);
    svms{k} = fitcsvm(trainingVector, class_k_label, 'Standardize',...
        true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', 10);
end

%% %**********************Classify the test data**********************
for k=1:NumOfClass
    if verbose
        fprintf(['Classifying with Classifier ', num2str(classes(k)),...
            ' of ', num2str(num_classes), '\n']);
    end
    [~, temp_score] = predict(svms{k}, hypervector);
    score(:, k) = temp_score(:, 2);                     
end
[~, est_label] = max(score, [], 2);
prediction_svm = im2uint8(zeros(h*w, 1));

for k=1:num_classes
    prediction_svm(find(est_label==k),:) = k;
end
prediction_svm = reshape(prediction_svm, [h, w, 1]);

z = find(hyperdata_gt == 0);
prediction_svm(z) = 0;


ERR = sum(sum( (prediction_svm ~= hyperdata_gt) ));
NumOfElements = sum(sum(NumOfClassElements(:)));
NumOfTrueElements = NumOfElements - ERR;
RATE = (NumOfTrueElements / NumOfElements)*100;

%% Results
fprintf(['\n','***********************************************************','\n']);
fprintf(['\t Success Rate  : ', num2str(RATE),'\n']);
fprintf(['***********************************************************','\n']);

%figure , imagesc(label2rgb(prediction_svm)) , title('Result');
%figure , imagesc(label2rgb(hyperdata_gt)) , title('Original');

[clsStat,mat_conf]=GetAccuracy(prediction_svm,hyperdata_gt);
disp(['Overall Accuracy:',num2str(clsStat.OA),', Kappa Coeffcient:', num2str(clsStat.Kappa)]);


% figure;
% subplot(1,2,1);imagesc(dataInfo.gt_data); 
% subplot(1,2,2);imagesc(prediction_svm);

figure('Name','SVM');
subplot(1,2,1); imagesc(label2rgb(prediction_svm)) , title('Result');
subplot(1,2,2); imagesc(label2rgb(hyperdata_gt)) , title('Original');
