function data = save_bb_masks(data)
%SAVE_BB_MASKS Saves bounding box information to 'de-register' the
%finished erosion, to properly overlap with GT.

    % Init bounding box variables
    if data.full_volume  % If the volume is the whole hand 
        data.joint1_bbs = false(size(data.im_bw));
        data.joint2_bbs = false(size(data.im_bw));
        data.joint3_bbs = false(size(data.im_bw));
        data.joint4_bbs = false(size(data.im_bw));
    else
        data.joint0_bbs = false(size(data.im_bw));
    end
    
     % Get bounding boxes for the two connected components in each joint
     if data.full_volume  % If the volume is the whole hand 
         rp1 = regionprops3(data.joint1_bw, 'BoundingBox');
         rp2 = regionprops3(data.joint2_bw, 'BoundingBox');
         rp3 = regionprops3(data.joint3_bw, 'BoundingBox');
         rp4 = regionprops3(data.joint4_bw, 'BoundingBox');
     else
         rp0 = regionprops3(data.joint0_bw, 'BoundingBox');
     end
    
    % Fill bounding box masks (2 CCs for each joint)
    if data.full_volume  % If the volume is the whole hand 
        data.joint1_bbs(ceil(rp1.BoundingBox(1,2)):ceil(rp1.BoundingBox(1,2))+ceil(rp1.BoundingBox(1,5)), ...
        ceil(rp1.BoundingBox(1,1)):ceil(rp1.BoundingBox(1,1))+ceil(rp1.BoundingBox(1,4)), ...
        ceil(rp1.BoundingBox(1,3)):ceil(rp1.BoundingBox(1,3))+ceil(rp1.BoundingBox(1,6))) = 1;
        data.joint1_bbs(ceil(rp1.BoundingBox(2,2)):ceil(rp1.BoundingBox(2,2))+ceil(rp1.BoundingBox(2,5)), ...
        ceil(rp1.BoundingBox(2,1)):ceil(rp1.BoundingBox(2,1))+ceil(rp1.BoundingBox(2,4)), ...
        ceil(rp1.BoundingBox(2,3)):ceil(rp1.BoundingBox(2,3))+ceil(rp1.BoundingBox(2,6))) = 1;
    
        data.joint2_bbs(ceil(rp2.BoundingBox(1,2)):ceil(rp2.BoundingBox(1,2))+ceil(rp2.BoundingBox(1,5)), ...
        ceil(rp2.BoundingBox(1,1)):ceil(rp2.BoundingBox(1,1))+ceil(rp2.BoundingBox(1,4)), ...
        ceil(rp2.BoundingBox(1,3)):ceil(rp2.BoundingBox(1,3))+ceil(rp2.BoundingBox(1,6))) = 1;
        data.joint2_bbs(ceil(rp2.BoundingBox(2,2)):ceil(rp2.BoundingBox(2,2))+ceil(rp2.BoundingBox(2,5)), ...
        ceil(rp2.BoundingBox(2,1)):ceil(rp2.BoundingBox(2,1))+ceil(rp2.BoundingBox(2,4)), ...
        ceil(rp2.BoundingBox(2,3)):ceil(rp2.BoundingBox(2,3))+ceil(rp2.BoundingBox(2,6))) = 1;
    
        data.joint3_bbs(ceil(rp3.BoundingBox(1,2)):ceil(rp3.BoundingBox(1,2))+ceil(rp3.BoundingBox(1,5)), ...
        ceil(rp3.BoundingBox(1,1)):ceil(rp3.BoundingBox(1,1))+ceil(rp3.BoundingBox(1,4)), ...
        ceil(rp3.BoundingBox(1,3)):ceil(rp3.BoundingBox(1,3))+ceil(rp3.BoundingBox(1,6))) = 1;
        data.joint3_bbs(ceil(rp3.BoundingBox(2,2)):ceil(rp3.BoundingBox(2,2))+ceil(rp3.BoundingBox(2,5)), ...
        ceil(rp3.BoundingBox(2,1)):ceil(rp3.BoundingBox(2,1))+ceil(rp3.BoundingBox(2,4)), ...
        ceil(rp3.BoundingBox(2,3)):ceil(rp3.BoundingBox(2,3))+ceil(rp3.BoundingBox(2,6))) = 1;
    
        data.joint4_bbs(ceil(rp4.BoundingBox(1,2)):ceil(rp4.BoundingBox(1,2))+ceil(rp4.BoundingBox(1,5)), ...
        ceil(rp4.BoundingBox(1,1)):ceil(rp4.BoundingBox(1,1))+ceil(rp4.BoundingBox(1,4)), ...
        ceil(rp4.BoundingBox(1,3)):ceil(rp4.BoundingBox(1,3))+ceil(rp4.BoundingBox(1,6))) = 1;
        data.joint4_bbs(ceil(rp4.BoundingBox(2,2)):ceil(rp4.BoundingBox(2,2))+ceil(rp4.BoundingBox(2,5)), ...
        ceil(rp4.BoundingBox(2,1)):ceil(rp4.BoundingBox(2,1))+ceil(rp4.BoundingBox(2,4)), ...
        ceil(rp4.BoundingBox(2,3)):ceil(rp4.BoundingBox(2,3))+ceil(rp4.BoundingBox(2,6))) = 1;
    
    else
        if size(rp0,1) > 0  % If at least one bone is present
            data.joint0_bbs(ceil(rp0.BoundingBox(1,2)):ceil(rp0.BoundingBox(1,2))+ceil(rp0.BoundingBox(1,5)), ...
            ceil(rp0.BoundingBox(1,1)):ceil(rp0.BoundingBox(1,1))+ceil(rp0.BoundingBox(1,4)), ...
            ceil(rp0.BoundingBox(1,3)):ceil(rp0.BoundingBox(1,3))+ceil(rp0.BoundingBox(1,6))) = 1;
        end
        if size(rp0,1) > 1  % If two bones are present
            data.joint0_bbs(ceil(rp0.BoundingBox(2,2)):ceil(rp0.BoundingBox(2,2))+ceil(rp0.BoundingBox(2,5)), ...
            ceil(rp0.BoundingBox(2,1)):ceil(rp0.BoundingBox(2,1))+ceil(rp0.BoundingBox(2,4)), ...
            ceil(rp0.BoundingBox(2,3)):ceil(rp0.BoundingBox(2,3))+ceil(rp0.BoundingBox(2,6))) = 1;
        end
    end
    
    % Ensure that bb's are still in the correct shape
    if data.full_volume  % If the volume is the whole hand 
        data.joint1_bbs = data.joint1_bbs(1:data.width,1:data.depth,1:data.n_slices); % crop bounding boxes
        data.joint2_bbs = data.joint2_bbs(1:data.width,1:data.depth,1:data.n_slices); % crop bounding boxes
        data.joint3_bbs = data.joint3_bbs(1:data.width,1:data.depth,1:data.n_slices); % crop bounding boxes
        data.joint4_bbs = data.joint4_bbs(1:data.width,1:data.depth,1:data.n_slices); % crop bounding boxes
    else
        data.joint0_bbs = data.joint0_bbs(1:data.width,1:data.depth,1:data.n_slices); % crop bounding boxes
    end
    
end