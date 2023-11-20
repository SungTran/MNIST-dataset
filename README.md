# MNIST-dataset
Solution to MNIST dataset

%Get data from csv files
data_Train1 = readtable('mnist_train1.csv');
data_Test = readtable('mnist_test.csv');
img_Train = table2array(data_Train1);
img_Test = table2array(data_Test);

%Split date into Predictors and Reponse
train_Predictor = img_Train(:,2:785);
test_Predictor = img_Test(:,2:785);
train_Response = categorical(img_Train(:,1));
test_Response =  categorical(img_Test(:,1));

%Create a 4D array to store images' features in Traing Dataset
c = rand(28, 28, 1, 60000);
for i = 1:60000
    a = train_Predictor(i,:);
    a = reshape(a,[28,28]);
    c(:,:,:,i) = a;
end

%Create a 4D array to store images' features in Testing Dataset
b = rand(28, 28, 1, 10000);
CONSTANT = zeros(28);
for i = 1:10000
    a = test_Predictor(i,:);
    a = reshape(a,[28,28]);
    b(:,:,:,i) = a;
end

%Define the structure and set up parameters of the CNN
layer = [
    imageInputLayer([28 28 1],'Name', 'Input')

    convolution2dLayer(3,8,'Padding','same','Name','Conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxPool_1')

    convolution2dLayer(3,16,'Padding','same','Name','Conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxPool_2')

    convolution2dLayer(3,32,'Padding','same','Name','Conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxPool_3')

    fullyConnectedLayer(10, 'Name', 'FC')
    softmaxLayer('Name', 'softMax')
    classificationLayer('Name', 'Output Classification')
];

%%Set up training parameters and train the network
options = trainingOptions("sgdm",'InitialLearnRate',0.01,'MaxEpochs', 4, 'Shuffle','every-epoch','Plots','training-progress');
net = trainNetwork(c, train_Response, layer, options);

%Calculate the accuracy of the trained network on testImages dataset.
YPred = classify(net,b);                                 
YValidation = test_Response;
accuracy = sum(YPred == YValidation)/numel(YValidation)
