%% Testing - Enter user input for Single person emotion recognition
clear vars emotionImageName;
testFolder = 'C:\Users\homeuser\Documents\MEJ\City MSc\05CV\Coursework\Code\Emotion\';
emotionImageName = 'IMG_3176[079]';
%% Interpret single image
emotionImageName = [sprintf('%s',emotionImageName) '.jpg'];
emotiontestImage = imread(fullfile(testFolder,emotionImageName));
emotionIndToTest = orientImage(emotiontestImage);

faceDetector = vision.CascadeObjectDetector;
BBOX = step(faceDetector, emotionIndToTest);
    
if(~isempty(BBOX))% contingency for no BBOX detected
    if(size(BBOX,1) > 1)% contingency for many BBOXes detected
        [~,N] = max(BBOX(:,3));
    else
        N = 1;% only one BBOX detected
    end

    x1 = round(0.08*BBOX(N,3));
    x2 = round(0.3*BBOX(N,3));
    y1 = round(0.5*BBOX(N,4));
    y2 = round(0.25*BBOX(N,4));

    xStart = BBOX(N,1)-x1;
    xEnd = BBOX(N,1)+BBOX(N,3)+x2;
    yStart = BBOX(N,2)-y1;
    yEnd = BBOX(N,2)+BBOX(N,4)+y2;

    if(xStart < 0 | yStart < 0 | xEnd > size(emotionIndToTest,2)| yEnd > size(emotionIndToTest,1))
        face = emotionIndToTest(BBOX(N,2):BBOX(N,2)+BBOX(N,4),BBOX(N,1):BBOX(N,1)+BBOX(N,3),:);
        imwrite(face,'DetectedFace.jpg');
    else
        face = emotionIndToTest(yStart:yEnd,xStart:xEnd,:);
        imwrite(face,'DetectedFace.jpg');
    end
    face = imresize(face,[300 300]);
    [hog_4x4, vis4x4] = extractHOGFeatures(face,'CellSize',[4 4]);
    model = loadCompactModel('HogSvm_Emotion');
    predictedLabel = predict(model, hog_4x4);
    disp('Emotion prediction done with HOG-SVM');
    fprintf('Image name is: %s and the Predicted label is: %d ',emotionImageName, predictedLabel);
else
    disp('No face detected');
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