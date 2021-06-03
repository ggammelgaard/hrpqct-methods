clear; clc; close all; diary off;  % Reset variables
opengl software  % Change volshow properties

% Filename
data.input_filename = 'Nobg_QTB024-2_MCP3';
%data.input_filename = '1_P282_S3_MCP3';
% data.input_filename = '1_P212_S2_MCP2';
% data.input_filename = 'Nobg_QTB024-2_MCP2';

% Load data
path_states = '..\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_detection.mat']);

%% Begin
close all;
% Orig image
BW = data.im_bw(:,:,153); 
figure(); imshow(BW);

% rprops = regionprops(img_data);
% img_data(uint8(rprops.Centroid(1)), uint8(rprops.Centroid(2))) = 0;
% figure(); imshow(img_data);

dim = size(BW);
col = round(dim(2)/2)-90;
row = min(find(BW(:,col)));
boundary = bwtraceboundary(BW,[row, col],'N');
figure();
imshow(ones(dim));
hold on
plot(boundary(:,2),boundary(:,1),'-k','LineWidth',2);

