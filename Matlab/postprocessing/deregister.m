function data = deregister(data)
%DEREGISTER Undo the registration on labelmap to ensure that prediction
%fits with GT.

    % 'Crop out' each joint with saved bounding boxes
    if data.full_volume  % If the volume is the whole hand
        joint1_overlap = data.joint1_bbs & data.erosions;
        joint2_overlap = data.joint2_bbs & data.erosions;
        joint3_overlap = data.joint3_bbs & data.erosions;
        joint4_overlap = data.joint4_bbs & data.erosions;
    else
        joint0_overlap = data.erosions;  % Overlap unecessary when only one joint
    end

    % Imwarp parameter
    cb_ref = imref2d(size(data.im_bw(:,:,1)));

    % Applying BOTTOM detransformation on each joint
    for i = data.base_slice_1+1:data.base_slice_2
        if data.full_volume  % If the volume is the whole hand     
            joint1_overlap(:,:,i) = imwarp(joint1_overlap(:,:,i), data.inv_tform01,'OutputView', cb_ref);
            joint2_overlap(:,:,i) = imwarp(joint2_overlap(:,:,i), data.inv_tform02,'OutputView', cb_ref);
            joint3_overlap(:,:,i) = imwarp(joint3_overlap(:,:,i), data.inv_tform03,'OutputView', cb_ref);
            joint4_overlap(:,:,i) = imwarp(joint4_overlap(:,:,i), data.inv_tform04,'OutputView', cb_ref);
        else
            joint0_overlap(:,:,i) = imwarp(joint0_overlap(:,:,i), data.inv_tform00,'OutputView', cb_ref);
        end
    end
    
    % Applying TOP detransformation on each joint
    for i = data.base_slice_2+1:data.n_slices  
        if data.full_volume  % If the volume is the whole hand     
            joint1_overlap(:,:,i) = imwarp(joint1_overlap(:,:,i), data.inv_tform11,'OutputView', cb_ref);
            joint2_overlap(:,:,i) = imwarp(joint2_overlap(:,:,i), data.inv_tform12,'OutputView', cb_ref);
            joint3_overlap(:,:,i) = imwarp(joint3_overlap(:,:,i), data.inv_tform13,'OutputView', cb_ref);
            joint4_overlap(:,:,i) = imwarp(joint4_overlap(:,:,i), data.inv_tform14,'OutputView', cb_ref);
        else
            joint0_overlap(:,:,i) = imwarp(joint0_overlap(:,:,i), data.inv_tform10,'OutputView', cb_ref);
        end
    end
    
    % Recombine joints into a full volume
    if data.full_volume  % If the volume is the whole hand   
        deregistered_erosions = logical(joint1_overlap+joint2_overlap+joint3_overlap+joint4_overlap);
    else
        deregistered_erosions = joint0_overlap;
    end
    
    % Update labelmap
    class = max(data.labelmap(:));  % Decided before this function
    data.labelmap = int8(deregistered_erosions)*class;
    
end