import importfile.*
import test.*

clear;
clc;

% Import command as struct with a vector per column
data = importfile('train.csv');

% Remove unnecessary columns
data = rmfield(data, 'A_listed_count');
data = rmfield(data, 'B_listed_count');

% Take ratio of follower count and set to new field
A_follower_count = data.A_follower_count;
A_following_count = data.A_following_count;
data = rmfield(data, 'A_follower_count');
data = rmfield(data, 'A_following_count');
for i = 1:size(A_following_count)
    if A_following_count(i) == 0
        A_following_count(i) = 1;
    end
end
A_follower_ratio = A_follower_count ./ A_following_count;
data.A_follower_ratio = A_follower_ratio;

B_follower_count = data.B_follower_count;
B_following_count = data.B_following_count;
data = rmfield(data, 'B_follower_count');
data = rmfield(data, 'B_following_count');
for i = 1:size(B_following_count)
    if B_following_count(i) == 0
        B_following_count(i) = 1;
    end
end
B_follower_ratio = B_follower_count ./ B_following_count;
data.B_follower_ratio = B_follower_ratio;

clear A_follower_count A_following_count A_follower_ratio
clear B_follower_count B_following_count B_follower_ratio

% Set new order for fields
fields = fieldnames(data);
new_order{18,1} = [];
new_order{1} = fields{18};
j = 2;
for i = 2:9
    new_order{j} = fields{i};
    j = j + 1;
end
new_order{j} = fields{19};
j = j + 1;
for i = 10:17
    new_order{j} = fields{i};
    j = j + 1;
end

% Remove labels to different vector
labels = data.Choice;
data = rmfield(data, 'Choice');

% Reorder struct
data = orderfields(data, new_order);

% Save data as a matrix
data_old = struct2array(data);

% Normalize data (
fields = fieldnames(data);
data.Fields = fields;
for i = 1:18
    data.(data.Fields{i}) = data.(data.Fields{i})/norm(data.(data.Fields{i}),2);
end
data = rmfield(data, 'Fields');

% Turn data to a matrix
data_normalized = struct2array(data);

clear new_order i j data