clear; clc; close all; diary off;  % Reset variables
opengl hardware  % Change volshow properties

% Filename
data.input_filename = 'Nobg_QTB024-2_MCP3';
% data.input_filename = '1_P282_S3_MCP3';
% data.input_filename = '1_P212_S2_MCP2';
% data.input_filename = 'Nobg_QTB024-2_MCP2';

% Load data
path_states = '.\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_preprocessing.mat']);

%% Begin
% Thresholding
data = thresholder(data);
img_thresh = data.im_bw(40:280,30:270,150); 
figure(); imshow(img_thresh);
% Morph ops
data = morph_ops(data);
img_morph = data.im_bw(40:280,30:270,150); 
figure(); imshow(img_morph);
% Active contour
data = active_contour(data);
img_ac = data.im_bw(40:280,30:270,150); 
figure(); imshow(img_ac);
