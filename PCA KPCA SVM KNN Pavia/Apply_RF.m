%% HIC 6 2021
clear; clc; close all;

%% Params
% User Params
%% Ratio(TrainData/Groundtruth)=0.01, 0.1, 0.3
perc_rate = 0.1;  

% trees in the forest
nTrees = 100;

%% Load Images
hyperdata = load('PaviaU.mat');
% hyperdata = hyperdata.salinas;
%hyperdata = hyperdata.indian_pines;
hyperdata = hyperdata.paviaU; 

hyperdata_gt = load('PaviaU_gt.mat');
% hyperdata_gt = hyperdata_gt.salinas_gt; 
%hyperdata_gt = hyperdata_gt.indian_pines_gt;
 hyperdata_gt = hyperdata_gt.paviaU_gt;

% figure; imshow(hyperdata);
%  imshow(label2rgb(hyperdata, @jet, [.5 .5 .5]))
%  
%   imshow(label2rgb(hyperdata_gt, @jet, [.5 .5 .5]))
% figure; imshow(hyperdata_gt);
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

%% Created Train and Label Vector %%etiket ve eðitimmm miii
[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
trainingVector = zeros(ClassSayisi,spec_size);
trainingVectorLabel = zeros(ClassSayisi,1); 

%% Training Vector & Training Vector Label

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train random forest

features = trainingVector(:,:);
classLabels = trainingVectorLabel(:,1);

% Train the TreeBagger (Decision Forest).
B = TreeBagger(nTrees,features,classLabels, 'Method', 'classification');
 
% Given a new individual WITH the features and WITHOUT the class label,
% what should the class label be? 
newData1 =  hypervector(:,:);

% Use the trained Decision Forest.
predChar1 = B.predict(newData1);

% Predictions is a char though. We want it to be a number.
predictedClass = str2double(predChar1);

% predictedClass_result = reshape(predictedClass,512,217); 
predictedClass_result = reshape(predictedClass,h,w); 

figure('Name','RF'); 
subplot(1,2,1); imagesc(label2rgb(predictedClass_result)); title('Result'); 
subplot(1,2,2); imagesc(label2rgb(hyperdata_gt));title('Groundtruth');
 
%% OBTAIN INFORMATION
  [m,n] = size(hyperdata_gt);
  CM = zeros(NumOfClass,NumOfClass);  %initialize the confusion matrix

%% OBTAIN THE CONFUSION MATRIX   
  for i=1:m
      for j=1:n
          if(hyperdata_gt(i,j)==0)
              continue;
          end
          t=predictedClass_result(i,j); %obtain the  label from the classification result
          k=hyperdata_gt(i,j); %obtain the true label
          CM(k,t)=CM(k,t)+1;   %confusion matrix assignment
      end
             
  end
%% CALCULATE EVALUATION METRICS
 [m,n] = size(CM);
  r=m;    %number of the classes
  rowsum=zeros(1,m);   %it is used to store the sum of the row value
  columnsum=zeros(1,n);  %it is used to store the sum of the column value
  N=0;   %it is used to store the total number of the pixels
  Diagsum=0; %sum of the diag
  ProAcc=zeros(m,1);  %store the producer accuracy for every class
  UserAcc=zeros(m,1); %store the user accuracy for every class
  AveAcc=zeros(2,1);  %store the average producer and user accuracy 
  for i=1:m                                                           
      for j=1:n
          rowsum(i)=rowsum(i)+CM(i,j); % sum of the rows and columns
          columnsum(j)=columnsum(j)+CM(i,j);
          N=N+CM(i,j);  % compute the total number of the pixels
          if(i==j)
              Diagsum=Diagsum+CM(i,i); % compute the sum of the pixels which are rightly classified
          end
      end
  end
  
  Pc=0;                                                              
  for i=1:r
      Pc=Pc+rowsum(i)*columnsum(i);
  end
 %% CALCULATE THE METRICS
  Kappa=(N*Diagsum-Pc)/(N*N-Pc);
  OveAcc=Diagsum/N;
 for i=1:m
      ProAcc(i)=CM(i,i)/rowsum(i);
     UserAcc(i)=CM(i,i)/columnsum(i);
 end
 %%
  AveAcc(1)=sum(ProAcc)/NumOfClass;
  AveAcc(2)=sum(UserAcc)/NumOfClass;

%% 
Kappa_result= Kappa*100;
%UserAcc_result = UserAcc * 100 ;
AveAcc_result2 = AveAcc(2)*100;
% AveAcc_result1= AveAcc(1)*100;
OveAcc_result = OveAcc *100;

disp(['Overall Accuracy:',num2str(OveAcc),', Kappa Coeffcient:', num2str(Kappa)]);
