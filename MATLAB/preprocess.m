import csvimport.*

table = csvimport('train.csv');
headers = table(1,:);
tags = table(2:end,1);
table = table(2:end,2:end);
tags = cell2mat(tags);
table = cell2mat(table);

test = csvimport('test.csv');
test_headers = test(1,:);
test = test(2:end,:);
test = cell2mat(test);
