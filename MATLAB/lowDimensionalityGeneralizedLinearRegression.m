GLMMdl = fitglm(table, tags);
ypredGLM = predict(GLMMdl, table);
ypredGLM = round(ypredGLM);
a = (tags == ypredGLM);
accuracyGeneralizedLinearRegression = sum(a(:) == 1)/5500 * 100;

% Accuracy also 71.6%