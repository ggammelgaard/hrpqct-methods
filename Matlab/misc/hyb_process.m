clear; clc; close all; diary off;  % Reset variables
opengl hardware  % Change volshow properties

% Filename
data.input_filename = 'Nobg_QTB024-2_MCP3';

% Load data
path_states = '..\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_detection.mat']);
% Load custom artifact
load('artifact.mat');

%% Begin
close all;
% Preprocessing
img_adj = im2uint8(data.im_adj(40:280,30:270,150)); 
img_adj = img_adj + 30;  % make a bit brighter
figure(); imshow(img_adj);

% Segmentation
img_bw = data.im_bw(40:280,30:270,150);
x = 52;
y = 183;
img_bw(x:x+14,y:y+14) = artifact;
figure(); imshow(img_bw);

% Registration
img_t = imtranslate(img_bw,[9, 5]);
figure(); imshowpair(img_bw, img_t)

% Volume cleaning
img_clean = data.im_bw(40:280,30:270,150);
figure(); imshow(img_clean);

% Erosion detection
% im_detection = cat(3, uint8(img_clean), uint8(img_clean), uint8(img_clean)).* 255;
im_detection = cat(3, img_adj,img_adj,img_adj);
erosion = logical(data.erosions(40:280,30:270,150));
for i = 1:size(im_detection,1)
    for j = 1:size(im_detection,1)
        if erosion(i,j) == 1
            im_detection(i,j,:) = [177, 122, 101];
        end
    end
end
figure(); imshow(im_detection);
