clear
clc

fprintf('Preprocessing data...\n')
preprocess;
fprintf('Done!\n');

lowDimensionalityLinearRegression;
fprintf('\nAccuracy with simple linear regression and tested on original dataset = %.3f%%\n', accuracyLinearRegression);

lowDimensionalityGeneralizedLinearRegression;
fprintf('\nAccuracy with generalized linear regression and tested on original dataset = %.3f%%\n', accuracyGeneralizedLinearRegression);

KNN;

fprintf('\nk-Nearest Neighbors classifier tested on original dataset with k = 71 => \n');
fprintf('\nAccuracy for simple Euclidean distance = %.3f%%\n', cell2mat(accuracyKNN(2,1)));
fprintf('\nAccuracy for inverse Euclidean distance = %.3f%%\n', cell2mat(accuracyKNN(2,2)));
fprintf('\nAccuracy for squared inverse Euclidean distance = %.3f%%\n', cell2mat(accuracyKNN(2,3)));
fprintf('\n');

fprintf('Cleaning up workspace...\n');
cleanup;