function main_single()
% Run hybrid detection method on one volume
restoredefaultpath;  % Reset paths
clear; clc; close all; diary off;  % Reset variables
% opengl hardware  % Change volshow properties

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
log_name = append(path_logs, datestr(now,'yymmdd_HHMMSS'), "_single.txt");
diary(log_name);
RAII.diary = onCleanup(@() diary('off'));

% Add meta data to data object
%data.input_filename = 'Nobg_QTB024-2_MCP2';
% data.input_filename = 'Nobg_QTB024-2_MCP3';
% data.input_filename = '1_P077_S3_MCP3'; 
% data.input_filename = '1_P103_S3_MCP2'; 
% data.input_filename = '1_P114_S1_MCP3';
% data.input_filename = '1_P204_S1_MCP3';
% data.input_filename = '1_P212_S2_MCP2';
% data.input_filename = '1_P212_S2_MCP2';
% data.input_filename  = '1_P040_S2_MCP3';
data.input_filename = '1_P017_S3_MCP2';

data.input_path = fullfile(pwd, '/../../../data/fm_data/');  % path to input folder
data.output_filename = data.input_filename;  
data.output_path = path_results;
data.full_volume = false;  % indicates if the input is a hand or a finger
data.wb = false; % decides if waitbars will be spawned in AC
data.parallel = true; % decides if parallelization should be applied

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

time = tic;  % start timer 

if 1
data = preprocessing(data);
save([path_states, data.input_filename, '_preprocessing.mat'], 'data', '-v7.3');
end

if 1
load([path_states, data.input_filename, '_preprocessing.mat']);
data = segmentation(data);
save([path_states, data.input_filename, '_segmentation.mat'], 'data', '-v7.3');
end

if 1
load([path_states, data.input_filename, '_segmentation.mat']);
data = registration(data);
save([path_states, data.input_filename, '_registration.mat'], 'data', '-v7.3');
end

if 1
load([path_states, data.input_filename, '_registration.mat']);
data = volume_cleaning(data);
save([path_states, data.input_filename, '_cleaning.mat'], 'data', '-v7.3');
end

if 1
load([path_states, data.input_filename, '_cleaning.mat']);
data = detection(data);
save([path_states, data.input_filename, '_detection.mat'], 'data', '-v7.3');
end

if 1
load([path_states, data.input_filename, '_detection.mat']);
data = generate_labelmap(data);
end

fprintf("main_single.m Script complete. Duration: %3.2f s\n", toc(time));
diary off

% opengl hardware
% figure(); volshow(logical(data.erosions));

end