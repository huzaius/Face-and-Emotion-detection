%% Clar all variables
clc;clear all;close all;tic;
%% Enter user input for Single person recognition
imagePath = 'C:\Users\homeuser\Documents\MEJ\City MSc\05CV\Coursework\FaceDetection\IndividualTest';
imageName = 'IMG_3150';
featureType = 'SURF';
classifierName = 'SVM';
%% Call to functions that perform face recognition
imageName = [sprintf('%s',imageName) '.jpg'];
I = imread(fullfile(imagePath,imageName));
faceDetails = RecogniseFace(I,featureType,classifierName);
%% FUNCTION DEFINITIONS
%% First point of call - Wrapper function to call other functions
function faceDetails = RecogniseFace(testImage,featureType,classifierType)
    
    [latestTestImage noOfFaces] = imageIsSingleOrGroup(testImage);
    disp(noOfFaces);
    if(noOfFaces == 0)
        disp('Not an image of person(s)');
        faceDetails = [];
        result = string('Label of photo is: %d, x-coordinate = %d, y-coordinate = %d');
        fprintf(result,faceDetails,faceDetails,faceDetails);        
    elseif(noOfFaces == 1)
        faceDetails = RecogniseSingleFace(latestTestImage,featureType,classifierType);%Single individual recognition   
        if ~strcmp(classifierType,'RF')
            result = string('Label of single photo is: %d, x-coordinate = %d, y-coordinate = %d');
            fprintf(result,faceDetails(1),faceDetails(2),faceDetails(3));
        else
            result = string('Label of single photo is: %s, x-coordinate = %d, y-coordinate = %d');
            fprintf(result,faceDetails{1},faceDetails{2},faceDetails{3});
        end
    elseif(noOfFaces > 1)
        faceDetails = RecogniseManyFaces(latestTestImage,featureType,classifierType);%Group recognition
        result = string('Face recognition on group photo done! %d faces identified');
        fprintf(result,size(faceDetails,1));
    end
end

%% Interpret image as single or group photo
function [img singleOrGroupPhoto] = imageIsSingleOrGroup(img)
    img = orientImage(img);
    faceDetector = vision.CascadeObjectDetector;
    BBOX = step(faceDetector, img);  
    
    if(size(BBOX,1) > 1)% contingency for many BBOXes detected
        [~,N] = max(BBOX(:,3));
        disp('Group photo detected');
    elseif(size(BBOX,1) == 1)
        N = 1;% only one BBOX detected
        disp('Single photo detected');
    else
        N = 0;
        disp('No faces detected');
    end
    singleOrGroupPhoto = N;    
end

%% Orient the image
function img = orientImage(img)
    faceDetector = vision.CascadeObjectDetector;
    BBOX = step(faceDetector, img);
    rotation = 0;    

    while(isempty(BBOX) & rotation ~= -360)
        rotation = rotation - 90;
        img = imrotate(img,rotation);
        BBOX = step(faceDetector, img);
    end    
end

%% Face recognition of single photo
function FaceDetails = RecogniseSingleFace(imageToTest,featureType,classifierType)
    [x, y] = size(imageToTest);
    comboFeatureClassifier = strcat(featureType,'-',classifierType);
      
    faceDetector = vision.CascadeObjectDetector;
    BBOX = step(faceDetector, imageToTest);
    if strcmp(classifierType,'RF')
        FaceDetails = {};
    end    
    if(~isempty(BBOX))% contingency for no BBOX detected
        if(size(BBOX,1) > 1)% contingency for many BBOXes detected
            [~,N] = max(BBOX(:,3));
        else
            N = 1;% only one BBOX detected
        end
        
        xFaceCenter = round(BBOX(N,1)+(BBOX(N,3)/2));% x-coordinate
        yFaceCenter = round(BBOX(N,2)+(BBOX(N,4)/2));% y-coordinate

            face = imageToTest(BBOX(N,2):BBOX(N,2)+BBOX(N,4),BBOX(N,1):BBOX(N,1)+BBOX(N,3),:);
    else
        disp('No face detected');
        predictedLabel = [];
        xFaceCenter = [];
        yFaceCenter = [];
        return;
    end    
    
    face = imresize(face,[300 300]);% retrieved face 
    switch comboFeatureClassifier% feature extraction for classification
        case 'HOG-SVM'
            [hog_4x4, vis4x4] = extractHOGFeatures(face,'CellSize',[4 4]);
            disp('HOG features extracted');
            model = loadCompactModel('HogSvm_i300x300_c4x4');
            predictedLabel = predict(model, hog_4x4);
            disp('HOG-SVM prediction complete');
        case 'HOG-RF'
            [hog_4x4, vis4x4] = extractHOGFeatures(face,'CellSize',[4 4]);
            disp('HOG features extracted');
            load('compactClassifierHogRF');
            predictedLabel = predict(compactClassifierHogRF,hog_4x4);
            disp('HOG-RF prediction complete');
        case 'HOG-FFNN'
            [hog_4x4, vis4x4] = extractHOGFeatures(face,'CellSize',[4 4]);
            load('HogPatternet.mat');
            prediction = netHog(hog_4x4');
            [~, predictedLabel] = max(prediction(:,1));
            disp('HOG-FFNN prediction complete')            
        case 'LBP-SVM'
            face = rgb2gray(face);
            LBPfeatures_1xN = extractLBPFeatures(face,'CellSize',[4 4]);
            disp('LBP features extracted');
            model = loadCompactModel('LbpSvm_i300x300_c4x4');
            predictedLabel = predict(model, LBPfeatures_1xN);
            disp('LBP-SVM prediction complete');
        case 'LBP-RF'
            face = rgb2gray(face);
            LBPfeatures_1xN = extractLBPFeatures(face,'CellSize',[4 4]);
            disp('LBP features extracted');
            load('compactClassifierLbpRF');
            predictedLabel = predict(compactClassifierLbpRF,LBPfeatures_1xN);
            disp('LBP-RF prediction complete');            
        case 'SURF-SVM'
            model = loadCompactModel('SurfSvm_i300x300_c4x4');
            structbag = load('bagOfFeature');
            bag = getfield(structbag,'bag');
            SURFfeatureVector  = encode(bag, face);
            predictedLabel = predict(model, SURFfeatureVector);        
        otherwise
            disp('Invalid combination chosen')
    end
    FaceDetails(1,1) = predictedLabel;
    if strcmp(classifierType,'RF')
        FaceDetails(1,2) = num2cell(xFaceCenter);
        FaceDetails(1,3) = num2cell(yFaceCenter);
    else
        FaceDetails(1,2) = xFaceCenter;
        FaceDetails(1,3) = yFaceCenter;
    end
end

%% Face recognition of group photo
function FaceDetails = RecogniseManyFaces(imageToTest,featureType,classifierType)
    comboFeatureClassifier = strcat(featureType,'-',classifierType);
    FaceDetector = vision.CascadeObjectDetector();
    BBOX = step(FaceDetector,imageToTest);
    if strcmp(classifierType,'RF')
        FaceDetails = {};
    end
    for j=1:size(BBOX,1)        
        xFaceCenter = round(BBOX(j,1)+(BBOX(j,3)/2));% x-coordinate
        yFaceCenter = round(BBOX(j,2)+(BBOX(j,4)/2));% y-coordinate        
        indFace = imageToTest(BBOX(j,2):BBOX(j,2)+BBOX(j,4),BBOX(j,1):BBOX(j,1)+BBOX(j,3),:);
        indFace = imresize(indFace,[300 300]);        
        switch comboFeatureClassifier% feature extraction for classification
            case 'HOG-SVM'
                [hog_4x4, vis4x4] = extractHOGFeatures(indFace,'CellSize',[4 4]);
                model = loadCompactModel('HogSvm_i300x300_c4x4');
                predictedLabel = predict(model, hog_4x4);
            case 'HOG-RF'
                [hog_4x4, vis4x4] = extractHOGFeatures(indFace,'CellSize',[4 4]);
                load('compactClassifierHogRF');
                predictedLabel = predict(compactClassifierHogRF,hog_4x4);
            case 'HOG-FFNN'
                [hog_4x4, vis4x4] = extractHOGFeatures(indFace,'CellSize',[4 4]);
                load('HogPatternet.mat');
                prediction = netHog(hog_4x4');
                [~, predictedLabel] = max(prediction(:,1));               
            case 'LBP-SVM'
                indFace = rgb2gray(indFace);
                LBPfeatures_1xN = extractLBPFeatures(indFace,'CellSize',[4 4]);
                model = loadCompactModel('LbpSvm_i300x300_c4x4');
                predictedLabel = predict(model, LBPfeatures_1xN);
            case 'LBP-RF'
                indFace = rgb2gray(indFace);
                LBPfeatures_1xN = extractLBPFeatures(indFace,'CellSize',[4 4]);
                load('compactClassifierLbpRF');
                predictedLabel = predict(compactClassifierLbpRF,LBPfeatures_1xN);             
            case 'SURF-SVM'
                model = loadCompactModel('SurfSvm_i300x300_c4x4');
                structbag = load('bagOfFeature');
                bag = getfield(structbag,'bag');
                SURFfeatureVector  = encode(bag, indFace);
                predictedLabel = predict(model, SURFfeatureVector);                          
            otherwise
                disp('Invalid combination chosen')
        end        
        FaceDetails(j,1) = predictedLabel;
        if strcmp(classifierType,'RF')
            FaceDetails(j,2) = num2cell(xFaceCenter);
            FaceDetails(j,3) = num2cell(yFaceCenter);
        else
            FaceDetails(j,2) = xFaceCenter;
            FaceDetails(j,3) = yFaceCenter;
        end
        imgCount = sprintf('Completed face:%d',j);
        disp(imgCount)
    end
        indFaceAnnotated = insertObjectAnnotation(imageToTest, 'rectangle', BBOX, FaceDetails(:,1));
        figure;
        imshow(indFaceAnnotated);    
end