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

% Morph ops
% Init vars
strel_size = 5;  % structuring elements
bw_size = 2000;
bw_tmp = data.im_bw;  % unecessary when dilate is not done on every slice

% Dilate
bw_tmp(:,:,1:data.base_slice_1) = imdilate(data.im_bw(:,:,1:data.base_slice_1), strel('sphere', strel_size));
bw_tmp(:,:,data.base_slice_1+1:data.base_slice_2) = imdilate(data.im_bw(:,:,data.base_slice_1+1:data.base_slice_2), strel('sphere', strel_size));
bw_tmp(:,:,data.base_slice_2+1:data.n_slices) = imdilate(data.im_bw(:,:,data.base_slice_2+1:data.n_slices), strel('sphere', strel_size));

img_dilate = bw_tmp(40:280,30:270,150); 
figure(); imshow(img_dilate);

% 2D fillling of closed bone structures on every slice
for i = 1:data.n_slices
    bw_tmp(:,:,i) = imfill(bw_tmp(:,:,i), 'holes');
end

img_fill = bw_tmp(40:280,30:270,150); 
figure(); imshow(img_fill);

% Erosion operation to negate the dilation effect on the surface of bone
data.im_bw(:,:,1:data.base_slice_1) = imerode(bw_tmp(:,:,1:data.base_slice_1), strel('sphere', strel_size));
data.im_bw(:,:,data.base_slice_1+1:data.base_slice_2) = imerode(bw_tmp(:,:,data.base_slice_1+1:data.base_slice_2), strel('sphere', strel_size));
data.im_bw(:,:,data.base_slice_2+1:data.n_slices) = imerode(bw_tmp(:,:,data.base_slice_2+1:data.n_slices), strel('sphere', strel_size));

img_erosion = data.im_bw(40:280,30:270,150); 
figure(); imshow(img_erosion);

% Remove small unconnected elements
for j = 1:data.n_slices
    data.im_bw(:,:,j) = bwareaopen(data.im_bw(:,:,j), bw_size);   
end

img_remove = data.im_bw(40:280,30:270,150); 
figure(); imshow(img_remove);