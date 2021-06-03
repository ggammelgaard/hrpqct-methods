clear; clc; close all; diary off;  % Reset variables
opengl hardware  % Change volshow properties

% Filename
data.input_filename  = '1_P040_S2_MCP3';

% Load data
path_states = '.\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_cleaning.mat']);

%% Begin
close all;
% Input
% cpos = [-4 0.5 0.65]*1.1;
rot = 90;
bw_rot = imrotate3(data.im_bw,rot,[1 0 0],'nearest','loose','FillValues',0);
bw_rot = imrotate3(bw_rot,-rot,[0 1 0],'nearest','loose','FillValues',0);
cpos_rot = [-1.15480710809239,0.895452278578039,6.00954122707289];  % found with export from "volumeViewer(mp_rot)"
cup_vector = [-0.0674568159156744,0.963386188377066,0.259493025013826];
figure(); volshow(bw_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);

% Watershed ignored, and delete range changed to (slice-3:slice+1)
data.im_clean = joint_splitter(data.im_bw);
clean_rot = imrotate3(data.im_clean,rot,[1 0 0],'nearest','loose','FillValues',0);
clean_rot = imrotate3(clean_rot,-rot,[0 1 0],'nearest','loose','FillValues',0);
figure(); volshow(clean_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);

data.im_clean = element_removal(data.im_clean, data.full_volume);
remov_rot = imrotate3(data.im_clean,rot,[1 0 0],'nearest','loose','FillValues',0);
remov_rot = imrotate3(remov_rot,-rot,[0 1 0],'nearest','loose','FillValues',0);
figure(); volshow(remov_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);

