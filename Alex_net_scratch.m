
clear
rng('shuffle')
    gpuDevice(1)

%url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';

downloadFolder='C:\Users\rrachako\Downloads\Matlab_DCNN_training\256_ObjectCategories';

%filename='Flower';
% Uncompressed data set
imageFolder = fullfile(downloadFolder,'256_ObjectCategories');


%if ~exist(imageFolder,'dir') % download only once
%    disp('Downloading Flower Dataset (218 MB)...');
%    websave(filename,url);
 %   untar(filename,downloadFolder)
%end

% Store the output in a temporary folder
%downloadFolder = tempdir;
%filename = fullfile(downloadFolder,'flower_dataset.tgz');


imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
    
[testDigitData,trainDigitData] = splitEachLabel(imds,0.3,'randomized');

trainDigitData.ReadFcn = @(filename)readAndPreprocessImage(filename);
testDigitData.ReadFcn = @(filename)readAndPreprocessImage(filename);
transferDigitData.ReadFcn = @(filename)readAndPreprocessImage(filename);

%net= alexnet;
net =importCaffeLayers('deploy_alexnet_places365.prototxt');

lgraph = layerGraph(net);

% 
% layersTransfer = net.Layers(1:end-3);
% 
% numClasses = 257;
% 
% 
% Layers = [layersTransfer
%     fullyConnectedLayer(numClasses,'Name','fc_last')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];

lgraph = removeLayers(lgraph, {'fc8','prob','output'});%discard output layers
numClasses = 257;%Set the fully connected layer to the same size as the number of classes in the new data sat. 
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc')%set the learning rate of new layers
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'drop7','fc'); %add the new output layers to the pretrained CNN
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)

rate2=[10e-4 10e-5  10e-6 10e-6 10e-4 10e-4 10e-5 10e-5 10e-5];



    miniBatchSize = 40;
validationFrequency = floor(numel(trainDigitData.Labels)/miniBatchSize);

options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.1,... 
      'LearnRateDropPeriod',3,... 
      'MaxEpochs',9,...
      'InitialLearnRate',rate2(1),...
      'MiniBatchSize',miniBatchSize);

    gpuDevice(1)
convnet = trainNetwork(trainDigitData,lgraph,options);



YPred = classify(convnet,testDigitData,'MiniBatchSize',100);
YTest = testDigitData.Labels;
test_accuracy = sum(YPred==YTest)/numel(YTest);


YPred = classify(convnet,trainDigitData,'MiniBatchSize',100);
YTest = trainDigitData.Labels;
training_accuracy = sum(YPred==YTest)/numel(YTest);


