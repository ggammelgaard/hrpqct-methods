function data = morph_ops(data)
%MORPH_OPS Closes contours and clean up the mask. (MO)
    time = tic;

    % Init vars
    strel_size = 4;  % 5; % structuring elements
    bw_size = 2000;
    bw_tmp = data.im_bw;  % unecessary when dilate is not done on every slice
    
    % Dilate
    bw_tmp(:,:,1:data.base_slice_1) = imdilate(data.im_bw(:,:,1:data.base_slice_1), strel('sphere', strel_size));
    bw_tmp(:,:,data.base_slice_1+1:data.base_slice_2) = imdilate(data.im_bw(:,:,data.base_slice_1+1:data.base_slice_2), strel('sphere', strel_size));
    bw_tmp(:,:,data.base_slice_2+1:data.n_slices) = imdilate(data.im_bw(:,:,data.base_slice_2+1:data.n_slices), strel('sphere', strel_size));
    
    % 2D fillling of closed bone structures on every slice
    for i = 1:data.n_slices
        bw_tmp(:,:,i) = imfill(bw_tmp(:,:,i), 'holes');
    end
    
    % Erosion operation to negate the dilation effect on the surface of bone
    data.im_bw(:,:,1:data.base_slice_1) = imerode(bw_tmp(:,:,1:data.base_slice_1), strel('sphere', strel_size));
    data.im_bw(:,:,data.base_slice_1+1:data.base_slice_2) = imerode(bw_tmp(:,:,data.base_slice_1+1:data.base_slice_2), strel('sphere', strel_size));
    data.im_bw(:,:,data.base_slice_2+1:data.n_slices) = imerode(bw_tmp(:,:,data.base_slice_2+1:data.n_slices), strel('sphere', strel_size));
    
    % Remove small unconnected elements
    for j = 1:data.n_slices
        data.im_bw(:,:,j) = bwareaopen(data.im_bw(:,:,j), bw_size);   
    end
    
    fprintf("%s Segmenting - Morphologic operations complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

