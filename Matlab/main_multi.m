function main_multi()
% Run hybrid detection method on multiple volumes
restoredefaultpath;  % Reset paths
clear; clc; close all; diary off; % Reset variables

% Add relevant paths
path_results = 'output/results/';  %  Create result folder
path_results = append(path_results, datestr(now,'yymmdd_HHMM'),'/');
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
log_name = append(path_logs, datestr(now,'yymmdd_HHMMSS'), "_multi.txt");
diary(log_name);
RAII.diary = onCleanup(@() diary('off'));

% Add meta data to data object
data.input_folder = fullfile(pwd, '/../../../data/skejby2020_4/');  % path to input folder
data.output_folder = path_results;
data.full_volume = false;  % indicate if the input is a hand or a finger
data.wb = false; % decides if waitbars will be spawned in AC
data.parallel = true;

% Kang params
data.kang_gamma = 2.5;
data.kang_beta = 8.5;
data.kang_alpha = 1;
data.kang_tp = 10000;

% Print params
data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% START PROCESS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_outer = tic;  % start timer

% Find all folders
subject_ids = dir([data.input_folder, 'P*']);
n_runs = size(subject_ids,1);
time_arr = zeros(1, n_runs);
for i = 1:n_runs
    fprintf("%s main_multi.m Run: %d of %d. Subject: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), i, n_runs, subject_ids(i).name);
    time_inner = tic;  % Start timer
    % Update data parameters
    data.input_path = [data.input_folder, subject_ids(i).name, '/'];
    data.input_filename = ['1_', subject_ids(i).name];
    data.output_path = [data.output_folder, subject_ids(i).name, '/'];
    
    % Run script
    data = preprocessing(data);
    data = segmentation(data);
    data = registration(data);
    data = volume_cleaning(data);
    data = detection(data);
    data = save_hrpqct(data);
    data = generate_labelmap(data);
    
    % Update time variable
    time_arr(i) = toc(time_inner);
    fprintf("%s main_multi.m Run %d complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), i, time_arr(i));
end

fprintf("\n%s main_multi.m Duration MEAN: %3.4f s, VAR: %3.4f s, data.parallel: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), mean(time_arr), var(time_arr), string(data.parallel));
fprintf("%s main_multi.m Script complete. Duration: %3.2f s\n\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time_outer));
diary off

end