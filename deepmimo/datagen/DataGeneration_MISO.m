% Script to construct a Mx2 MIMO multicell scenario batch from a DeepMIMO
% Dataset. Splits scenarios into training and test samples.
% Requires DeepMIMO_Dataset_Generator from the DeepMIMO source in folder.
[DeepMIMO_dataset,params]=DeepMIMO_Dataset_Generator();

folder = 'DeepMIMO Dataset';
scenario_name = 'scenario2';

num_rows = params.active_user_last - params.active_user_first +1;
num_user_pos = length(DeepMIMO_dataset{1}.user); % make it even
width = num_user_pos / num_rows;
num_users_area12 = floor(num_rows/2) * width;
num_users_area34 = num_user_pos - num_users_area12;

trainingdata_setsize = 5000;
testdata_setsize = 1000;
K = 4; % always 4 due to area split
I = 16;
M = 12;

% split into training and testdata
% training: even users
% test: odd users
% Furthermore, split coverage areas of BSs

num_half_users_area12 = floor(num_users_area12 / 2);
num_half_users_area34 = floor(num_users_area34 / 2);

training.channels = cell(1, K);
training.user_idx = zeros(trainingdata_setsize, I);
test.channels = cell(1, K);
test.user_idx = zeros(testdata_setsize, I);

% Sampling
for i = 1:trainingdata_setsize
    training.user_idx(i, 1:(I/2)) = randsample(num_half_users_area12, I/2) * 2;
    training.user_idx(i, (I/2+1):I) = randsample(num_half_users_area34, I/2) * 2 + num_users_area12;
end
for i = 1:testdata_setsize
    test.user_idx(i, 1:(I/2)) = randsample(num_half_users_area12, I/2) * 2 - 1;
    test.user_idx(i, (I/2+1):I) = randsample(num_half_users_area34, I/2) * 2 - 1 + num_users_area12;
end

% training.user_idx = cast(training.user_idx, 'int64');
% test.user_idx = cast(test.user_idx, 'int64');

for i_bs = 1:K
    training.channels{i_bs} = cell(1, I);
    test.channels{i_bs} = cell(1, I);
    for i_user = 1: I
        % Training
        H = zeros(trainingdata_setsize, 1, M);
        for i_sample = 1:trainingdata_setsize
            H(i_sample, :, :) = DeepMIMO_dataset{i_bs}.user{training.user_idx(i_sample, i_user)}.channel.';
        end
        training.channels{i_bs}{i_user} = H;
        
        % Test
        H = zeros(testdata_setsize, 1, M);
        for i_sample = 1:testdata_setsize
            H(i_sample, :, :) = DeepMIMO_dataset{i_bs}.user{test.user_idx(i_sample, i_user)}.channel.';
        end
        test.channels{i_bs}{i_user} = H;
    end
end

save(fullfile(folder, [scenario_name, '_training']), '-struct', 'training');
save(fullfile(folder, [scenario_name, '_test']), '-struct', 'test');