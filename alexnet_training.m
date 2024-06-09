%---------------
clear all; close all; clc;


% Load the training and test data
trainDataPath = 'E:\Revised_AI_Paper_CQT\Datbase\RALE_SOUNDSLIBARARY\CQT_Images\train';
testDataPath = 'E:\Revised_AI_Paper_CQT\Datbase\RALE_SOUNDSLIBARARY\CQT_Images\test';

% Create an image datastore for training and testing
trainData = imageDatastore(trainDataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

testData = imageDatastore(testDataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Define the size expected by AlexNet
inputSize = [227 227];

% Create augmented image datastores to automatically resize images
augTrainData = augmentedImageDatastore(inputSize, trainData);
augTestData = augmentedImageDatastore(inputSize, testData);

% Define the number of classes
numClasses = numel(categories(trainData.Labels));

% Load pre-trained AlexNet
net = alexnet;

% Modify the network for transfer learning
layersTransfer = net.Layers(1:end-3);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augTestData, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
trainedNet = trainNetwork(augTrainData, layers, options);

% Evaluate the network
predictedLabels = classify(trainedNet, augTestData);
accuracy = mean(predictedLabels == testData.Labels);

fprintf('Test accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix and classification report for AlexNet
confMatAlexNet = confusionmat(testData.Labels, predictedLabels);
disp('Confusion Matrix for AlexNet:');
disp(confMatAlexNet);

% Calculate precision, recall, F1 score for AlexNet
precisionAlexNet = diag(confMatAlexNet)./sum(confMatAlexNet,2);
recallAlexNet = diag(confMatAlexNet)./sum(confMatAlexNet,1)';
f1scoreAlexNet = 2 * (precisionAlexNet .* recallAlexNet) ./ (precisionAlexNet + recallAlexNet);

% Display classification report for AlexNet
fprintf('Classification Report for AlexNet:\n');
fprintf('Class\tPrecision\tRecall\tF1 Score\n');
for i = 1:numel(precisionAlexNet)
    fprintf('%s\t%.2f\t\t%.2f\t%.2f\n', string(categories(testData.Labels(i))), ...
        precisionAlexNet(i)*100, recallAlexNet(i)*100, f1scoreAlexNet(i)*100);
end

% Plot confusion matrices
figure;
confusionchart(confMatAlexNet);
title('Confusion Matrix for AlexNet');
