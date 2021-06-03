clear; clc; close all; diary off;  % Reset variables
opengl hardware  % Change volshow properties

% Filename
data.input_filename  = '1_P017_S3_MCP2';

% Load data
path_states = '.\output\debug_states\'; 
addpath(path_states);
load([path_states, data.input_filename, '_cleaning.mat']);

%% Begin
close all;

% Init map for each bone
m = false(size(data.im_clean));
p = false(size(data.im_clean));

% Find all connected components
cc = bwconncomp(data.im_clean);
% Fill maps with each bone
p(cc.PixelIdxList{1}) = 1;
m(cc.PixelIdxList{2}) = 1;

% Shave off 20 slices nearest to the MCP joint 
rp = regionprops3(p);
p(:,:,round(max((rp.BoundingBox(1,6)- 20 ),1)):round(rp.BoundingBox(1,6))) = 0;
rp = regionprops3(m);
m(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+ 20 ),data.n_slices))) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%
mp = logical(p+m);
rot = 60;
mp_rot = imrotate3(mp,rot,[1 0 0],'nearest','loose','FillValues',0);
cpos_rot = [2.65315082807626,-1.86706744612945,5.26543918729311]*0.55;  % found with export from "volumeViewer(mp_rot)"
cup_vector = [0.125490283016572,0.730980969125542,0.670760025377709];
figure(); volshow(mp_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 1 - Find breaks that are detectable in 2 dimensions
p_breaks = object_finder(p, data.n_slices, data.base_slice_1, data.base_slice_2);
fprintf('%s Detection - p_breaks\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
m_breaks = object_finder(m, data.n_slices, data.base_slice_1, data.base_slice_2);
fprintf('%s Detection - m_breaks\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
% breaks_combined = logical(p_breaks + m_breaks);
breaks_combined = m_breaks;

%%%%%%%%%%%%%%%%%%%%%%%%%%
breaks_combined_rot = imrotate3(breaks_combined,rot,[1 0 0],'nearest','loose','FillValues',0);
figure(); volshow(breaks_combined_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 2 - Determine the cortical border
% Combine original image with found breaks to create complete cortical bone
full_structure  = logical(logical(m+p) + breaks_combined);
full_structure  = imclose(full_structure, strel('sphere',1));
% Cortical border
cortical = imerode(full_structure, strel('sphere',4));
cortical = logical(imabsdiff(full_structure,cortical));

%%%%%%%%%%%%%%%%%%%%%%%%%%
cortical_rot = imrotate3(cortical,rot,[1 0 0],'nearest','loose','FillValues',0);
figure(); volshow(cortical_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 3 - Isolate erosions
% Find breaks that goes all the way through the cortical border to the
% trabecular bone
trabecular_breaks = imsubtract(breaks_combined, cortical);
trabecular_breaks(trabecular_breaks == -1) = 0;
% Removes voxels that does not have any neighbours in their 3x3x3 area
trabecular_breaks = bwmorph3(trabecular_breaks,'clean');
% Removes voxels that does not have more than 14 neighbours in their 3x3x3 area
trabecular_breaks = bwmorph3(trabecular_breaks,'majority');
% Remove components that are smaller than 100 voxels
trabecular_breaks = bwareaopen(trabecular_breaks, 100);

% Add back the part of the erosion that is part of the cortical border
erosions = imdilate(trabecular_breaks, strel('sphere',4));  % "Close operation"
erosions = imsubtract(logical(erosions), logical(data.im_clean));  
erosions(erosions == -1) = 0;
erosions = bwareaopen(erosions, 100);  % Just some cleaning it seems

%%%%%%%%%%%%%%%%%%%%%%%%%%
erosions_rot = imrotate3(erosions,rot,[1 0 0],'nearest','loose','FillValues',0);
figure(); volshow(erosions_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 4 - Erosion refinement
% Check if detected erosions are part of larger erosion
if data.full_volume  % If the volume is the whole hand 
    erosions = find_connected_objects(erosions,logical(p2+p3+p4+p5), data.n_slices);
    erosions = find_connected_objects(erosions,logical(m2+m3+m4+m5), data.n_slices);
else
    erosions = find_connected_objects(erosions,logical(p), data.n_slices);
    erosions = find_connected_objects(erosions,logical(m), data.n_slices);
end

% Remove erosions found on the outside of the bone surface
erosions = remove_outer_objects(erosions, data.im_clean);

%%%%%%%%%%%%%%%%%%%%%%%%%%
erosions_rot = imrotate3(logical(erosions),rot,[1 0 0],'nearest','loose','FillValues',0);
figure(); volshow(erosions_rot, 'Backgroundcolor', [1,1,1], 'CameraPosition', cpos_rot, 'CameraUpVector', cup_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%