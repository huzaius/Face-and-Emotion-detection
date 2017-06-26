clc;clear all;close all;tic;
%% Preparing dataset for training and testing
rng(100);
inputFolder = 'C:\Users\homeuser\Documents\MEJ\City MSc\05CV\Coursework\';
rootFolder = fullfile(inputFolder, 'BboxDataSet');
imgSets = [ imageSet(fullfile(rootFolder, '01')), imageSet(fullfile(rootFolder, '02')),...
imageSet(fullfile(rootFolder, '03')), imageSet(fullfile(rootFolder, '04')), imageSet(fullfile(rootFolder, '05')), imageSet(fullfile(rootFolder, '06')),... 
imageSet(fullfile(rootFolder, '07')), imageSet(fullfile(rootFolder, '08')), imageSet(fullfile(rootFolder, '09')), imageSet(fullfile(rootFolder, '10')), imageSet(fullfile(rootFolder, '11')), imageSet(fullfile(rootFolder, '12')),... 
imageSet(fullfile(rootFolder, '13')), imageSet(fullfile(rootFolder, '14')), imageSet(fullfile(rootFolder, '15')), ...
imageSet(fullfile(rootFolder, '16')), imageSet(fullfile(rootFolder, '17')), imageSet(fullfile(rootFolder, '18')), imageSet(fullfile(rootFolder, '19')), imageSet(fullfile(rootFolder, '20')), imageSet(fullfile(rootFolder, '21')), imageSet(fullfile(rootFolder, '22')), imageSet(fullfile(rootFolder, '23')), imageSet(fullfile(rootFolder, '24')), imageSet(fullfile(rootFolder, '25')), imageSet(fullfile(rootFolder, '26')), ...
imageSet(fullfile(rootFolder, '27')), imageSet(fullfile(rootFolder, '28')), imageSet(fullfile(rootFolder, '29')), imageSet(fullfile(rootFolder, '30')), imageSet(fullfile(rootFolder, '31')), imageSet(fullfile(rootFolder, '32')), imageSet(fullfile(rootFolder, '33')), imageSet(fullfile(rootFolder, '34')), ...
imageSet(fullfile(rootFolder, '35')), imageSet(fullfile(rootFolder, '36')), imageSet(fullfile(rootFolder, '37'))];
[trainSet, testSet] = partition(imgSets, 0.85, 'randomize');
noOfLabels = length(trainSet);
noOfImages = 0;
for i = 1: length(trainSet)
    noOfImages = noOfImages + trainSet(i).Count;
end
noOfTrainImages = noOfImages;noOfImages = 0;
for i = 1: length(testSet)
    noOfImages = noOfImages + testSet(i).Count;
end
noOfTestImages = noOfImages;
%% Feature 1 : HOG feature extraction
[trainHOGFeatures, trainHOGLabels] = trainSetHOGfeatures(trainSet);
[testHOGFeatures, testHOGLabels] = testSetHOGfeatures(testSet);

%% Model 1a: SVM training for HOG features
tic;classifierHogSvm = fitcecoc(trainHOGFeatures, trainHOGLabels);time(1)= toc;
disp('HOG-SVM classifier creation is done');
predictedHogSvmTrainLabels = predict(classifierHogSvm, trainHOGFeatures);
predictedHogSvmTestLabels = predict(classifierHogSvm, testHOGFeatures);
disp('HOG-SVM prediction complete');
[confMatHogSvmTrain orderHogSvmTrain] = confusionmat(trainHOGLabels, predictedHogSvmTrainLabels);
[confMatHogSvmTest orderHogSvmTest] = confusionmat(testHOGLabels, predictedHogSvmTestLabels);
% HogSvmCV = crossval(classifierHogSvm,'Kfold',5);
% HogSvmKloss = kfoldLoss(HogSvmCV,'mode','individual');
% disp('HOG-SVM performance evaluation done');
saveCompactModel(classifierHogSvm,'HogSvm_i300x300_c4x4');
disp('HOG-SVM CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 2a : Random  Forest training with HOG features
tic;classifierHogRF = TreeBagger(50,trainHOGFeatures,trainHOGLabels,'OOBPrediction','On', 'Method','classification');time(2)= toc;
predictedHogRFTestLabels = predict(classifierHogRF,testHOGFeatures);
[confMatHogRFTest orderHogRFTest] = confusionmat(testHOGLabels, PredictedHogRFTestLabels);
oobErrorHogRF = oobError(classifierHogRF,'Mode','cumulative');
plot(1:50,oobErrorHogRF);xlabel('No. of trees');ylabel('OOB Error');
disp('HOG-RF performance evaluation done');
compactClassifierHogRF = compact(classifierHogRF);
save compactClassifierHogRF;
disp('HOG-RF CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 3a : Neural network training with HOG features
xTrain = zeros(size(trainHOGFeatures,2),noOfTrainImages);k = 1;
for i = 1:noOfTrainImages
    xTrain(:,i) = trainHOGFeatures(i,:);
end
xTest = zeros(size(testHOGFeatures,2),noOfTestImages);k = 1;
for i = 1:noOfTestImages
    xTest(:,i) = testHOGFeatures(i,:);
end
disp('Train and test HOG feature vectorisation for Neural nets done');

xTrainLabels = zeros(noOfLabels,noOfTrainImages);
for i = 1 : noOfTrainImages
    xTrainLabels(trainHOGLabels(i),i) = 1;
end
xTestLabels = zeros(noOfLabels,noOfTestImages);
for i = 1 : noOfTestImages
    xTestLabels(testHOGLabels(i),i) = 1;
end
disp('Train and test HOG label formating for Neural nets done');

netHog = patternnet(100, 'trainscg');
netHog.divideParam.trainRatio = 90/100;
netHog.divideParam.valRatio = 10/100;
netHog.divideParam.testRatio = 0;
netHog = configure(netHog,xTrain,xTrainLabels);
tic;netHog = train(netHog,xTrain, xTrainLabels);time(3)= toc;
view(netHog);
disp('HOG-Neural net classifier creation done');

trainPrediction = netHog(xTrain);
for i = 1 : noOfTrainImages
    [~, predictedHogNNTrainLabels(i)] = max(trainPrediction(:,i));
end
testPrediction = netHog(xTest);
for i = 1 : noOfTestImages
    [~, predictedHogNNTestLabels(i)] = max(testPrediction(:,i));
end
disp('HOG-Neural net prediction done');

ACCTrain = sum(predictedHogNNTrainLabels == trainHOGLabels) / noOfTrainImages;
ACCTest = sum(predictedHogNNTestLabels == testHOGLabels) / noOfTestImages;%% Train a classifier using the extracted features
[confMatHogNNTrain orderHogNNTrain] = confusionmat(trainHOGLabels, predictedHogNNTrainLabels);
[confMatHogNNTest orderHogfNNTest] = confusionmat(testHOGLabels, predictedHogNNTestLabels);
disp('HOG-Neural net performance evaluation done');
save('HogPatternet.mat','netHog');
% [bestNetFromCV,bestParamFromCV,AvgCrossValidationMSE] = CrossValidationFFN(5,netHog,xTrain,xTrainLabels)
disp('HOG-NN CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 4a : Naive Bayes training with HOG features
classifierHogNB = fitcnb(trainHOGFeatures, trainHOGLabels);
% HogNBCV = crossval(classifierHogNB);
% HogNBKloss = kfoldLoss(HogNBCV);
PredictedHogNBTestLabels = predict(classifierHogNB,testHOGFeatures);
[confMatHogNBTest orderHogNBTest] = confusionmat(testHOGLabels, PredictedHogNBTestLabels);
disp('HOG-NB CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Feature 2 : SURF feature extraction
clear vars kStart kEnd trainSURFLabels;
bag = bagOfFeatures(trainSet);
TrainSURFfeatureMatrix  = encode(bag, trainSet);
kStart = 0;
for i = 1: length(trainSet)
    kEnd = kStart + trainSet(i).Count;
    for j = kStart+1:kEnd
        trainSURFLabels(j,1) = i;
    end
    kStart = 0;
    for h = 1:i
    kStart = kStart + trainSet(h).Count;
    end
end
disp('Train set SURF feature extraction done');

bagTest = bagOfFeatures(testSet);
TestSURFfeatureMatrix  = encode(bag, testSet);
clear vars kStart kEnd testSURFLabels;
kStart = 0;
for i = 1: length(testSet)
    kEnd = kStart + testSet(i).Count;
    for j = kStart+1:kEnd
        testSURFLabels(j,1) = i;
    end
    kStart = 0;
    for h = 1:i
    kStart = kStart + testSet(h).Count;
    end
end
disp('Test set SURF feature extraction done');

% Plot the histogram of visual word occurrences
img = read(imgSets(1), 5);
featureVector = encode(bagOfSURFtrainFeatures, img);
figure; bar(featureVector)
title('Visual word occurrences for Label 5 "Greg"')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

%% Model 1b: SVM training with SURF features
tic;classifierSurfSvm = fitcecoc(trainSURFFeatures, trainSURFLabels);time(4)= toc;
disp('SURF-SVM classifier creation is done');
predictedSurfSvmTrainLabels = predict(classifierSurfSvm, trainSURFFeatures);
predictedSurfSvmTestLabels = predict(classifierSurfSvm, TestSURFfeatureMatrix);
disp('SURF-SVM prediction complete');
[confMatSurfSvmTrain orderSurfSvmTrain] = confusionmat(trainSURFLabels, predictedSurfSvmTrainLabels);
[confMatSurfSvmTest orderSurfSvmTest] = confusionmat(testSURFLabels, predictedSurfSvmTestLabels);
% SurfSvmCV = crossval(classifierSurfSvm,'Kfold',5);
% SurfSvmKloss = kfoldLoss(SurfSvmCV,'mode','individual');
% disp('SURF-SVM performance evaluation done');
saveCompactModel(classifierSurfSvm,'SurfSvm_i300x300_c4x4');
disp('SURF-SVM CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');
plotconfusion(testSURFLabels,predictedSurfSvmTestLabels);
%% Model 2b : Random  Forest training with SURF features
tic;classifierSurfRF = TreeBagger(100,trainSURFFeatures,trainSURFLabels,'OOBPrediction','On', 'Method','classification');time(5)= toc;
PredictedSurfRFTestLabels = predict(classifierSurfRF,TestSURFfeatureMatrix);
oobErrorSurfRF = oobError(classifierSurfRF,'Mode','cumulative');
plot(oobErrorSurfRF);
% [confMatSurfRFTest orderSurfRFTest] = confusionmat(testSURFLabels, predictedSurfRFTestLabels);
disp('SURF-RF performance evaluation done');
compactClassifierSurfRF = compact(classifierSurfRF);
save compactClassifierSurfRF;
disp('SURF-RF CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 3b : Neural network training with SURF features
xTrain = zeros(size(trainSURFFeatures,2),noOfTrainImages);k = 1;
for i = 1:noOfTrainImages
    xTrain(:,i) = trainSURFFeatures(i,:);
end
xTest = zeros(size(TestSURFfeatureMatrix,2),noOfTestImages);k = 1;
for i = 1:noOfTestImages
    xTest(:,i) = TestSURFfeatureMatrix(i,:);
end
disp('Train and test SURF feature vectorisation for Neural nets done');

xTrainLabels = zeros(noOfLabels,noOfTrainImages);
for i = 1 : noOfTrainImages
    xTrainLabels(trainSURFLabels(i),i) = 1;
end
xTestLabels = zeros(noOfLabels,noOfTestImages);
for i = 1 : noOfTestImages
    xTestLabels(testSURFLabels(i),i) = 1;
end
disp('Train and test SURF label formating for Neural nets done');

netSurf = patternnet(100, 'trainscg');
netSurf.divideParam.trainRatio = 90/100;
netSurf.divideParam.valRatio = 10/100;
netSurf.divideParam.testRatio = 0;
netSurf = configure(netSurf,xTrain,xTrainLabels);
tic;netSurf = train(netSurf,xTrain, xTrainLabels);time(6)= toc;
view(netSurf);
disp('SURF-Neural net classifier creation done');

trainPrediction = netSurf(xTrain);
for i = 1 : noOfTrainImages
    [~, predictedSurfNNTrainLabels(i)] = max(trainPrediction(:,i));
end
testPrediction = netSurf(xTest);
for i = 1 : noOfTestImages
    [~, predictedSurfNNTestLabels(i)] = max(testPrediction(:,i));
end
disp('SURF-Neural net prediction done');

ACCTrain = sum(predictedSurfNNTrainLabels == trainSURFLabels) / noOfTrainImages;
ACCTest = sum(predictedSurfNNTestLabels == testSURFLabels) / noOfTestImages;%% Train a classifier using the extracted features
[confMatSurfNNTrain orderSurfNNTrain] = confusionmat(trainSURFLabels, predictedSurfNNTrainLabels);
[confMatSurfNNTest orderSurfNNTest] = confusionmat(testSURFLabels, predictedSurfNNTestLabels);
disp('SURF-Neural net performance evaluation done');
save netSurf;
disp('SURF-NN CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 4b : Train Naive Bayes with SURF features
% classifierSurfNB = fitcnb(trainSURFFeatures, trainSURFLabels);
% SurfNBCV = crossval(classifierSurfNB);
% SurfNBKloss = kfoldLoss(SurfNBCV);
% PredictedSurfNBTestLabels = predict(classifierSurfNB,TestSURFfeatureMatrix);
% [confMatSurfNBTest orderSurfNBTest] = confusionmat(testSURFLabels, PredictedSurfNBTestLabels);
% disp('SURF-NB CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Feature 3 : LBP feature extraction
[trainLBPFeatures, trainLBPLabels] = trainSetLBPfeatures(trainSet);
[testLBPFeatures, testLBPLabels] = testSetLBPfeatures(testSet);

%% Model 1c: SVM training for LBP features
tic;classifierLbpSvm = fitcecoc(trainLBPFeatures, trainLBPLabels);time(7)= toc;
disp('LBP-SVM classifier creation is done');
predictedLbpSvmTrainLabels = predict(classifierLbpSvm, trainLBPFeatures);
predictedLbpSvmTestLabels = predict(classifierLbpSvm, testLBPFeatures);
disp('LBP-SVM prediction complete');
[confMatLbpSvmTrain orderLbpSvmTrain] = confusionmat(trainLBPLabels, predictedLbpSvmTrainLabels);
[confMatLbpSvmTest orderLbpSvmTest] = confusionmat(testLBPLabels, predictedLbpSvmTestLabels);
% LbpSvmCV = crossval(classifierLbpSvm,'Kfold',5);
% LbpSvmKloss = kfoldLoss(LbpSvmCV,'mode','individual');
% disp('LBP-SVM performance evaluation done');
saveCompactModel(classifierLbpSvm,'LbpSvm_i300x300_c4x4');
disp('LBP-SVM CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 2c : Random  Forest training with LBP features
tic;classifierLbpRF = TreeBagger(50,trainLBPFeatures,trainLBPLabels,'OOBPrediction','On', 'Method','classification');time(8)= toc;
predictedLbpRFTestLabels = predict(classifierLbpRF,testLBPFeatures);
oobErrorLbpRF = oobError(classifierLbpRF,'Mode','cumulative');
plot(1:50,oobErrorLbpRF);
[confMatLbpRFTest orderLbpRFTest] = confusionmat(testLBPLabels, PredictedLbpRFTestLabels);
disp('LBP-RF performance evaluation done');
compactClassifierLbpRF = compact(classifierLbpRF);
save compactClassifierLbpRF;
disp('LBP-RF CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 3c : Neural network training with LBP features
xTrain = zeros(size(trainLBPFeatures,2),noOfTrainImages);k = 1;
for i = 1:noOfTrainImages
    xTrain(:,i) = trainLBPFeatures(i,:);
end
xTest = zeros(size(testLBPFeatures,2),noOfTestImages);k = 1;
for i = 1:noOfTestImages
    xTest(:,i) = testLBPFeatures(i,:);
end
disp('Train and test LBP feature vectorisation for Neural nets done');

xTrainLabels = zeros(noOfLabels,noOfTrainImages);
for i = 1 : noOfTrainImages
    xTrainLabels(trainLBPLabels(i),i) = 1;
end
xTestLabels = zeros(noOfLabels,noOfTestImages);
for i = 1 : noOfTestImages
    xTestLabels(testLBPLabels(i),i) = 1;
end
disp('Train and test LBP label formating for Neural nets done');

netLbp = patternnet(100, 'trainscg');
netLbp.divideParam.trainRatio = 90/100;
netLbp.divideParam.valRatio = 10/100;
netLbp.divideParam.testRatio = 0;
netLbp = configure(netLbp,xTrain,xTrainLabels);
tic;netLbp = train(netLbp,xTrain, xTrainLabels);time(9)= toc;
view(netLbp);
disp('LBP-Neural net classifier creation done');

trainPrediction = netLbp(xTrain);
for i = 1 : noOfTrainImages
    [~, predictedLbpNNTrainLabels(i)] = max(trainPrediction(:,i));
end
testPrediction = netLbp(xTest);
for i = 1 : noOfTestImages
    [~, predictedLbpNNTestLabels(i)] = max(testPrediction(:,i));
end
disp('LBP-Neural net prediction done');

ACCTrain = sum(predictedLbpNNTrainLabels == trainLBPLabels) / noOfTrainImages;
ACCTest = sum(predictedLbpNNTestLabels == testLBPLabels) / noOfTestImages;%% Train a classifier using the extracted features
[confMatLbpNNTrain orderLbpNNTrain] = confusionmat(trainLBPLabels, predictedLbpNNTrainLabels);
[confMatLbpNNTest orderLbpNNTest] = confusionmat(testLBPLabels, predictedLbpNNTestLabels);
disp('LBP-Neural net performance evaluation done');
save netLbp;
disp('LBP-NN CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');

%% Model 4c : Naive Bayes training with LBP features
% classifierLbpNB = fitcnb(trainLBPFeatures, trainLBPLabels);
% LbpNBCV = crossval(classifierLbpNB);
% LbpNBKloss = kfoldLoss(LbpNBCV);
% PredictedLbpNBTestLabels = predict(classifierLbpNB,testLBPFeatures);
% [confMatLbpNBTest orderLbpNBTest] = confusionmat(testLBPLabels, PredictedLbpNBTestLabels);
% disp('LBP-NB CLASSIFIER TRAINING AND EVALUATION ARE COMPLETE');