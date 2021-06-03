function exp_kang_bone()
% Run hybrid detection method on multiple volumes
clear; clc; close all; diary off; % Reset variables
cd ..  % Change current folder to the one above

% Add relevant paths
path_results = 'output/results/';  %  Create result folder
if exist(path_results, 'dir') ~= 7  
    mkdir(path_results);
end
path_states = 'output/debug_states/';  %  Create debug folder
if exist(path_states, 'dir') ~= 7  
    mkdir(path_states);
end
path_logs = 'output/logs/';  %  Create log folder
if exist(path_logs, 'dir') ~= 7  
    mkdir(path_logs);
end
addpath(path_results);
addpath(path_states);
addpath(path_logs);
addpath('preprocessing')
addpath('segmentation')
addpath('registration')
addpath('volume_cleaning')
addpath('detection')
addpath('postprocessing')

% Start logfile
log_name = append(path_logs, datestr(now,'yymmdd_HHMMSS'), "_exp_kang_bone.txt");
diary(log_name);
RAII.diary = onCleanup(@() diary('off'));

% Add meta data to data object
data.input_folder = fullfile(pwd, '/../../../data/skejby2020_4/');  % path to input folder
data.output_folder = path_results;
data.full_volume = false;  % indicate if the input is a hand or a finger
data.wb = false; % decides if waitbars will be spawned in AC
data.parallel = true;

% Define params
gamma =  [0, 2, 4];
beta = [0, 5, 10];
alpha = [-1, 0, 1];
tp = [10000]; % (0:5000:15000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% START PROCESS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_outer = tic;  % start timer

% Make param matrix and score array
param_matrix = combvec(gamma,beta,alpha,tp);
score_arr = zeros(1, size(param_matrix,2));
n_combinations = length(score_arr);

% Find all folders
subject_ids = dir([data.input_folder, 'P*']);
% n_files = size(subject_ids,1);
n_files = 48;  % 10 percent of data

fprintf("%s exp_kang_bone.m n_combinations: %d\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), n_combinations);
fprintf("%s exp_kang_bone.m n_files: %d\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), n_files);
for i = 1:n_combinations
    % Init DICE vars
    tp_sum = 0;
    fp_sum = 0;
    fn_sum = 0;
    % Update Kang params:
    data.kang_gamma = param_matrix(1,i);
    data.kang_beta = param_matrix(2,i);
    data.kang_alpha = param_matrix(3,i);
    data.kang_tp = param_matrix(4,i);

    for j = 1:n_files
        fprintf("%s exp_kang_bone.m Run: %d of %d. ParamCombination: %d, Subject: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), ((i-1)*n_files+j), n_files*n_combinations, i, subject_ids(j).name);
        time_inner = tic;  % Start timer
        
        % Update data parameters
        data.input_path = [data.input_folder, subject_ids(j).name, '/'];
        data.input_filename = ['1_', subject_ids(j).name];

        % Compute thresholder bone map
        data = preprocessing(data);
        data = thresholder(data);
        bone_pred = data.im_bw;
        % Load GT bone map
        bone_true = load_bone_map(data);
        % Calc tp, fp, and fn score
        tmp_tp = sum(bone_true(:) .* bone_pred(:));
        tmp_fn = sum(bone_true(:)) - tmp_tp;
        tmp_tn = sum(~bone_true(:) .* ~bone_pred(:));
        tmp_fp = sum(~bone_true(:)) - tmp_tn;
        % Update ouside vars
        tp_sum = tp_sum + tmp_tp;
        fp_sum = fp_sum + tmp_fp;
        fn_sum = fn_sum + tmp_fn;

%         opengl hardware
%         figure(); volshow(bone_pred);

        fprintf("%s exp_kang_bone.m Run %d complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), ((i-1)*n_files+j), toc(time_inner));
    end

    % Calc dice
    score_arr(i) = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum);

end

% Save results
output_path = [data.output_folder, datestr(now,'yymmdd_HHMM')];
save([output_path, '_score_arr.mat'], 'score_arr');
save([output_path, '_param_matrix.mat'], 'param_matrix');
fprintf("%s exp_kang_bone.m Saved score as: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), [output_path, '_score_arr.mat']);
fprintf("%s exp_kang_bone.m Saved params as: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), [output_path, '_param_matrix.mat']);

[val, idx] = max(score_arr);
fprintf("%s exp_kang_bone.m Best combination idx: %d , score: %3.4f\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), idx, val);
fprintf("%s exp_kang_bone.m kang_gamma: %3.2f\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), param_matrix(1,idx));
fprintf("%s exp_kang_bone.m kang_beta: %3.2f\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), param_matrix(2,idx));
fprintf("%s exp_kang_bone.m kang_alpha: %3.2f\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), param_matrix(3,idx));
fprintf("%s exp_kang_bone.m kang_tp: %3.2f\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), param_matrix(4,idx));
   
score_arr  % print just so it is more than one place
fprintf("%s exp_kang_bone.m Script complete. Duration: %3.2f s\n\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time_outer));
diary off

end

function bone_map = load_bone_map(data)
    data_path    = [data.input_path, '2' ,data.input_filename(2:end), '_label.nii.gz'];
    label_map = int8(niftiread(data_path));  % load tensor
    bone_map = (label_map == 1);
end


