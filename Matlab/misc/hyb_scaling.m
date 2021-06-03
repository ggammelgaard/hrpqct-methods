clear; clc; close all; diary off;  % Reset variables
opengl hardware  % Change volshow properties

% Filename
% data.input_filename = 'Nobg_QTB024-2_MCP3';
data.input_filename = '1_P282_S3_MCP3';
% data.input_filename = '1_P212_S2_MCP2';
% data.input_filename = 'Nobg_QTB024-2_MCP2';

% Load data
path_states = '..\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_detection.mat']);

%% Begin
close all;
% Orig image
img_data = data.im_data(50:250,50:250,116); 
img_data = img_data + (40/2^8)*2^16;  % make a bit brighter
figure(); imshow(img_data);

% Scaled image
img_adj = data.im_adj(50:250,50:250,116); 
img_adj = img_adj + (30/2^8)*2^16;  % make a bit brighter
figure(); imshow(img_adj);

% Prepare orig volume
tmp_data = uint16(zeros(size(data.im_data)));
% Convert data from int16 to uint16
for i = 1:data.n_slices  
    tmp_data(:,:,i) = im2uint16(data.im_data(:,:,i));  
end

% Histograms
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
edges = 0:600:2^16;

purp = '#882e72';
purp2 = '#994F88';
blue = '#377eb8';
oran = '#e8601c';
grey1 = '#46353A';
grey2 = '#777777';
grey3 = '#C0C0C0';
black = '#000000';
color = purp2;

f1=figure(); histogram(tmp_data, edges,'FaceColor',color,'Linestyle','none','FaceAlpha',1)
xlim([-1000 2^16+1000]);
ylim([0 2000000]);
xlabel('Voxel intensity value')
ylabel('Number of voxels')
set(gca,'FontSize',12)
set(f1,'Position',[680   558   560*1.3   420])

f2 = figure(); histogram(data.im_adj, edges,'FaceColor',color,'Linestyle','none','FaceAlpha',1)
xlim([-1000 2^16+1000]);
ylim([0 200000]);
xlabel('Voxel intensity value')
ylabel('Number of voxels')
set(gca,'FontSize',12)
set(f2,'Position',[680   558   560*1.3   420])