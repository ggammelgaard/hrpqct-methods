function [bw_3d_diff, bw_3d] = object_finder(image, slices, base_slice_1, base_slice_2, parallel)
%OBJECT_FINDER Find breaks that are detectable in 2 dimensions (objectFinder2)

% Init local variables
[X,Y,Z] = size(image);
strel_size = 20;
% base_slice_1 = 108;  % Slice that separates section 1&2
% base_slice_2 = 218;  % Slice that separates section 2&3

% Init map for each volume section
tmp_sec1_init = false(size(image));
tmp_sec2_init = false(size(image));
tmp_sec3_init = false(size(image));
% Fill each map with the corresponding section
tmp_sec1_init(:,:,1:base_slice_1) = image(:,:,1:base_slice_1);
tmp_sec2_init(:,:,base_slice_1+1:base_slice_2) = image(:,:,base_slice_1+1:base_slice_2);
tmp_sec3_init(:,:,base_slice_2+1:slices) = image(:,:,base_slice_2+1:Z);

% Perform 2D operation in XY axis
tmp_sec1 = tmp_sec1_init;
tmp_sec2 = tmp_sec2_init;
tmp_sec3 = tmp_sec3_init;
if parallel
    parfor i = 1:Z
       tmp_sec1(:,:,i) = imclose(tmp_sec1(:,:,i), strel('disk', strel_size));
       tmp_sec2(:,:,i) = imclose(tmp_sec2(:,:,i), strel('disk', strel_size));
       tmp_sec3(:,:,i) = imclose(tmp_sec3(:,:,i), strel('disk', strel_size));
    end
else
    for i = 1:Z
       tmp_sec1(:,:,i) = imclose(tmp_sec1(:,:,i), strel('disk', strel_size));
       tmp_sec2(:,:,i) = imclose(tmp_sec2(:,:,i), strel('disk', strel_size));
       tmp_sec3(:,:,i) = imclose(tmp_sec3(:,:,i), strel('disk', strel_size));
    end
end
bwz = logical(tmp_sec1+tmp_sec2+tmp_sec3);  % Axial plane

% Perform 2D operation in XZ axis
tmp_sec1 = tmp_sec1_init;
tmp_sec2 = tmp_sec2_init;
tmp_sec3 = tmp_sec3_init;
if parallel
    parfor i = 1:Y
       tmp_sec1(:,i,:) = imclose(squeeze(tmp_sec1(:,i,:)), strel('disk', strel_size));
       tmp_sec2(:,i,:) = imclose(squeeze(tmp_sec2(:,i,:)), strel('disk', strel_size));
       tmp_sec3(:,i,:) = imclose(squeeze(tmp_sec3(:,i,:)), strel('disk', strel_size));
    end
else 
    for i = 1:Y
       tmp_sec1(:,i,:) = imclose(squeeze(tmp_sec1(:,i,:)), strel('disk', strel_size));
       tmp_sec2(:,i,:) = imclose(squeeze(tmp_sec2(:,i,:)), strel('disk', strel_size));
       tmp_sec3(:,i,:) = imclose(squeeze(tmp_sec3(:,i,:)), strel('disk', strel_size));
    end
end
bwy = logical(tmp_sec1+tmp_sec2+tmp_sec3);  % Coronal plane


% Perform 2D operation in YZ axis
tmp_sec1 = tmp_sec1_init;
tmp_sec2 = tmp_sec2_init;
tmp_sec3 = tmp_sec3_init;
if parallel
    parfor i = 1:X
       tmp_sec1(i,:,:) = imclose(squeeze(tmp_sec1(i,:,:)), strel('disk', strel_size));
       tmp_sec2(i,:,:) = imclose(squeeze(tmp_sec2(i,:,:)), strel('disk', strel_size));
       tmp_sec3(i,:,:) = imclose(squeeze(tmp_sec3(i,:,:)), strel('disk', strel_size));
    end
else
    for i = 1:X
       tmp_sec1(i,:,:) = imclose(squeeze(tmp_sec1(i,:,:)), strel('disk', strel_size));
       tmp_sec2(i,:,:) = imclose(squeeze(tmp_sec2(i,:,:)), strel('disk', strel_size));
       tmp_sec3(i,:,:) = imclose(squeeze(tmp_sec3(i,:,:)), strel('disk', strel_size));
    end
end
bwx = logical(tmp_sec1+tmp_sec2+tmp_sec3);  % Sagittal plane

% Combined map (not used)
bw_3d = logical(bwz+bwy+bwx);

% Difference between results and original image
bwz_diff = imabsdiff(bwz, image);
bwy_diff = imabsdiff(bwy, image);
bwx_diff = imabsdiff(bwx, image);

% Create closed binary image
bw_3d_diff = bwz_diff.*bwy_diff + bwz_diff.*bwx_diff + bwy_diff.*bwx_diff;
% Removes voxels that does not have any neighbours in their 3x3x3 area
bw_3d_diff = bwmorph3(bw_3d_diff,'clean');
% Removes voxels that does not have more than 14 neighbours in their 3x3x3 area
bw_3d_diff = bwmorph3(bw_3d_diff,'majority');

end

