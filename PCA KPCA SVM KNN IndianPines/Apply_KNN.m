%% HIC 6 2021
clear; clc; close all;
%% Params

%% Ratio(TrainData/Groundtruth)=0.01, 0.1, 0.3
perc_rate = 0.01;   
%% knn_number = input('The number of K ?: ')
knn_number=3;
%% Load Images
hyperdata = load('indian_pines.mat');
% hyperdata = hyperdata.salinas;
hyperdata = hyperdata.indian_pines;
%hyperdata = hyperdata.paviaU; 


hyperdata_gt = load('indian_pines_gt.mat');
% hyperdata_gt = hyperdata_gt.salinas_gt; 
hyperdata_gt = hyperdata_gt.indian_pines_gt;
%hyperdata_gt = hyperdata_gt.paviaU_gt;
%% Normalization
%hyperdata = ( hyperdata - min(min(min(hyperdata))) ) /  ( max(max(max(hyperdata))) - min(min(min(hyperdata))) );
[h,w,spec_size] = size(hyperdata);
% data2vector
hypervector = reshape(hyperdata , [h*w,spec_size]);

%% 	Variables

%Data for training 
Mask = zeros(size(hyperdata_gt));  

% Number of classes in hyper image 
NumOfClass = max(hyperdata_gt(:)); 

% Variable to hold the number of tags for each class
NumOfClassElements = zeros(1,NumOfClass);        
%% Calculate each class's elements number
for i = 1 : 1 : NumOfClass
    NumOfClassElements(i) = double(sum(sum((hyperdata_gt == i)))); 
end
%% ** %Perc_Rate data will select for training vector ** 
for i = 1 : 1 : NumOfClass
    perc_5 = floor(NumOfClassElements(i) * perc_rate);
    [row,col] = find(hyperdata_gt == i);
    
    while(sum(sum(Mask == i)) ~= perc_5) 
        x = floor((rand() * (NumOfClassElements(i) - 1)) + 1);
        Mask(row(x),col(x)) = hyperdata_gt(row(x),col(x));
    end
end

%% Class's Nums

ClassSayisi = 0;
for i = 1 : 1 : NumOfClass
   ClassSayisi = ClassSayisi + sum(sum(Mask == i)); 
end

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

%% Created Train and Label Vector 

[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
% already holds row, col and labels of values that are not 0 in mask 
trainingVector = zeros(ClassSayisi,spec_size); 
trainingVectorLabel = zeros(ClassSayisi,1);

%% Training Vector & Training Vector Label  

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train knn 
distance=0; 
counter=size(trainingVector,1); %Number of Total Train Data 
neighbors=zeros(1,knn_number); 
tagged=zeros(h,w); %Classified Image  
 for x=1:h %Calculating Euclidean Distance 
  for y=1:w           
      for z=1:counter             
          for band=1:spec_size         
              distance = distance + (hyperdata(x,y,band)- hyperdata(trainingData_row(z,1),trainingData_col(z,1),band))^2;  
          end
          dist(z,1)=sqrt(distance);  
          distance=0;    
      end
      [v , index]=sort(dist(:,1));     
      for k=1:knn_number      
          neighbors(k)=trainingVectorLabel(index(k),1);    
      end
  tagged(x,y)=mode(neighbors); % Selecting the most frequent class tag               
  end
 end
 
 figure('Name','KNN'); 
 subplot(1,2,1); imagesc(tagged); title('KNN Classified Hyperspectral Image'); 
 subplot(1,2,2); imagesc(hyperdata_gt); title('Groundtruth');   
 
 %% for knn
 
  %CALCULATING CLASSIFICATION SUCCESS RATE 
 true_positive=0;
 false_positive=0;  

 for x=1:h       
     for y=1:w    
         if(hyperdata_gt(x,y)~=0)            
             if(tagged(x,y) == hyperdata_gt(x,y))       
                 true_positive = true_positive+1; 
              else
                 false_positive = false_positive+1;
             end
         end
     end
 end
 
%success_rate= true_positive*100 / (true_positive + false_positive); 
%print=['Success Rate of KNN Classification = %',num2str(success_rate)]; 
%disp(print)

[clsStat,mat_conf]=GetAccuracy(tagged,hyperdata_gt);
disp(['Overall Accuracy:',num2str(clsStat.OA),', Kappa Coeffcient:', num2str(clsStat.Kappa)]);


