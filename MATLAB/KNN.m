KNNMdl = fitcknn(table, tags, 'BreakTies', 'random', 'KFold', 15, 'NumNeighbors', 71);

yKNN = double.empty(0,5500);
%--------------------------------------------------------------------------

avg_accuracy1 = 0;
for i = 1:15
    labels = predict(KNNMdl.Trained{i}, table);
    a = (tags == labels);
    labels = labels';
    yKNN = [yKNN; labels];
    accuracy = sum(a(:) == 1)/5500 * 100;
    avg_accuracy1 = avg_accuracy1 + accuracy;
end
avg_accuracy1 = avg_accuracy1 / 15;

% Accuracy = 75.65%
%--------------------------------------------------------------------------

KNNMdl2 = fitcknn(table, tags, 'BreakTies', 'random', 'KFold', 15, 'DistanceWeight', 'inverse', 'NumNeighbors', 71);
avg_accuracy2 = 0;
for i = 1:15    
    labels = predict(KNNMdl2.Trained{i}, table);
    a = (tags == labels);
    labels = labels';
    yKNN = [yKNN; labels];
    accuracy = sum(a(:) == 1)/5500 * 100;
    avg_accuracy2 = avg_accuracy2 + accuracy;
end
avg_accuracy2 = avg_accuracy2 / 15;

% Accuracy = 97.59%
%--------------------------------------------------------------------------

KNNMdl3 = fitcknn(table, tags, 'BreakTies', 'random', 'KFold', 15, 'DistanceWeight', 'squaredinverse', 'NumNeighbors', 71);
avg_accuracy3 = 0;
for i = 1:15    
    labels = predict(KNNMdl3.Trained{i}, table);
    a = (tags == labels);
    labels = labels';
    yKNN = [yKNN; labels];
    accuracy = sum(a(:) == 1)/5500 * 100;
    avg_accuracy3 = avg_accuracy3 + accuracy;
end
avg_accuracy3 = avg_accuracy3 / 15;

% Accuracy = 97.55%
%--------------------------------------------------------------------------

colNames = {'equal_distance', 'inverse_distance', 'squared_inverse_distance'};
accuracyKNN = [avg_accuracy1 avg_accuracy2 avg_accuracy3];
accuracyKNN = [colNames;num2cell(accuracyKNN)];