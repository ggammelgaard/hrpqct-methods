function data = detection(data)
%DETECTION Detects erosions and create logical map with them (DetectionMethod)
    time = tic;  % start timer
    
    % Init map for each bone
    if data.full_volume  % If the volume is the whole hand  
        m2 = false(size(data.im_clean));
        m3 = false(size(data.im_clean));
        m4 = false(size(data.im_clean));
        m5 = false(size(data.im_clean));
        p2 = false(size(data.im_clean));
        p3 = false(size(data.im_clean));
        p4 = false(size(data.im_clean));
        p5 = false(size(data.im_clean));
    else
        m = false(size(data.im_clean));
        p = false(size(data.im_clean));
    end
        
    % Find all connected components
    cc = bwconncomp(data.im_clean);
    if cc.NumObjects < 1  % If no bones are present
        fprintf('%s Detection - ERROR! No bones has been found.\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        fprintf("%s Detection complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
        data.erosions = zeros(size(data.im_clean)); % no erosions found
        return;
    end
    
    % Fill maps with each bone
    if data.full_volume  % If the volume is the whole hand  
        p2(cc.PixelIdxList{4}) = 1;
        p3(cc.PixelIdxList{3}) = 1;
        p4(cc.PixelIdxList{2}) = 1;
        p5(cc.PixelIdxList{1}) = 1;
        m2(cc.PixelIdxList{5}) = 1;
        m3(cc.PixelIdxList{6}) = 1;
        m4(cc.PixelIdxList{7}) = 1;
        m5(cc.PixelIdxList{8}) = 1;
    else
        p(cc.PixelIdxList{1}) = 1;
        if cc.NumObjects > 1  % If two bones are present
            m(cc.PixelIdxList{2}) = 1;
        end
    end
    
    % Shave off 20 slices nearest to the MCP joint 
    if data.full_volume  % If the volume is the whole hand 
        rp = regionprops3(p2);
        p2(:,:,round(max((rp.BoundingBox(1,6)-20),1)):round(rp.BoundingBox(1,6))) = 0;
        rp = regionprops3(p3);
        p3(:,:,round(max((rp.BoundingBox(1,6)-20),1)):round(rp.BoundingBox(1,6))) = 0;
        rp = regionprops3(p4);
        p4(:,:,round(max((rp.BoundingBox(1,6)-20),1)):round(rp.BoundingBox(1,6))) = 0;
        rp = regionprops3(p5);
        p5(:,:,round(max((rp.BoundingBox(1,6)-20),1)):round(rp.BoundingBox(1,6))) = 0;
        rp = regionprops3(m2);
        m2(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+20),data.n_slices))) = 0;
        rp = regionprops3(m3);
        m3(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+20),data.n_slices))) = 0;
        rp = regionprops3(m4);
        m4(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+20),data.n_slices))) = 0;
        rp = regionprops3(m5);
        m5(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+20),data.n_slices))) = 0; 
    else
        rp = regionprops3(p);
        p(:,:,round(max((rp.BoundingBox(1,6)- 20 ),1)):round(rp.BoundingBox(1,6))) = 0;
        if cc.NumObjects > 1  % If two bones are present
            rp = regionprops3(m);
            m(:,:,round(rp.BoundingBox(1,3)):round(min((rp.BoundingBox(1,3)+ 20 ),data.n_slices))) = 0;
        else
            fprintf('%s Detection - WARNING! Only one bone has been found.\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        end
    end

    % Step 1 - Find breaks that are detectable in 2 dimensions
    if data.full_volume  % If the volume is the whole hand 
        m2_breaks = object_finder(m2, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - M2 (1/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        m3_breaks = object_finder(m3, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - M3 (2/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        m4_breaks = object_finder(m4, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - M4 (3/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        m5_breaks = object_finder(m5, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - M5 (4/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        p2_breaks = object_finder(p2, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - P2 (5/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        p3_breaks = object_finder(p3, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - P3 (6/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        p4_breaks = object_finder(p4, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - P4 (7/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        p5_breaks = object_finder(p5, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - P5 (8/8)\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        breaks_combined = logical(m2_breaks+m3_breaks+m4_breaks+m5_breaks+p2_breaks+p3_breaks+p4_breaks+p5_breaks);
    else
        p_breaks = object_finder(p, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - p_breaks\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        m_breaks = object_finder(m, data.n_slices, data.base_slice_1, data.base_slice_2, data.parallel);
        fprintf('%s Detection - m_breaks\n', datestr(now,'yyyy-mm-dd HH:MM:SS'))
        breaks_combined = logical(p_breaks + m_breaks);
    end
    
    % Step 2 - Determine the cortical border
    % Combine original image with found breaks to create complete cortical bone
    full_structure  = logical(data.im_clean + breaks_combined);
    full_structure  = imclose(full_structure, strel('sphere',1));
    % Cortical border
    cortical = imerode(full_structure, strel('sphere',4));
    cortical = logical(imabsdiff(full_structure,cortical));
    
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
    
    % Save to data
    data.erosions = erosions;

    fprintf("%s Detection complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
end

