LMMdl = fitlm(table, tags);
ypredLM = predict(LMMdl, table);
ypredLM = round(ypredLM);
a = (tags == ypredLM);
accuracyLinearRegression = sum(a(:) == 1)/5500 * 100;

% Accuracy = 71.6%