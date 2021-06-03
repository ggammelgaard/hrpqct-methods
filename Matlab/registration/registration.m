function data = registration(data)
%REGISTRATION Aligns each joint separately to fix registration issue
    time = tic;

    % Crop out each joint (only relevant for hand volumes)
    if data.full_volume  % If the volume is the whole hand
        data = crop_joints(data);
    else
        data.joint0_bw = data.im_bw;
        data.joint0 = uint16(zeros(size(data.im_bw)));
    end
    
    % Parameters for finding transformations
    cb_ref = imref2d(size(data.im_bw(:,:,1)));
    params.GradientMagnitudeTolerance = 0.0001;
    params.MinimumStepLength = 0.00001;
    params.MaximumStepLength = 0.0325;
    params.MaximumIterations = 100;
    params.RelaxationFactor = 0.5;
    
    % Finding optimal transformation for BOTTOM registration on each joint
    if data.full_volume  % If the volume is the whole hand     
        [tform01,~] = register(double(data.joint1_bw(:,:,data.base_slice_1+1)), double(data.joint1_bw(:,:,data.base_slice_1)),1,params);
        [tform02,~] = register(double(data.joint2_bw(:,:,data.base_slice_1+1)), double(data.joint2_bw(:,:,data.base_slice_1)),1,params);
        [tform03,~] = register(double(data.joint3_bw(:,:,data.base_slice_1+1)), double(data.joint3_bw(:,:,data.base_slice_1)),1,params);
        [tform04,~] = register(double(data.joint4_bw(:,:,data.base_slice_1+1)), double(data.joint4_bw(:,:,data.base_slice_1)),1,params);
    else
        [tform00,~] = register(double(data.joint0_bw(:,:,data.base_slice_1+1)), double(data.joint0_bw(:,:,data.base_slice_1)),1,params);    
    end
    
    % Applying BOTTOM transformation on each joint
    for i = data.base_slice_1+1:data.base_slice_2
        if data.full_volume  % If the volume is the whole hand     
            data.joint1_bw(:,:,i) = imwarp(data.joint1_bw(:,:,i), tform01,'OutputView', cb_ref);
            data.joint1(:,:,i) = imwarp(data.joint1(:,:,i), tform01,'OutputView', cb_ref);
            data.joint2_bw(:,:,i) = imwarp(data.joint2_bw(:,:,i), tform02,'OutputView', cb_ref);
            data.joint2(:,:,i) = imwarp(data.joint2(:,:,i), tform02,'OutputView', cb_ref);
            data.joint3_bw(:,:,i) = imwarp(data.joint3_bw(:,:,i), tform03,'OutputView', cb_ref);
            data.joint3(:,:,i) = imwarp(data.joint3(:,:,i), tform03,'OutputView', cb_ref);
            data.joint4_bw(:,:,i) = imwarp(data.joint4_bw(:,:,i), tform04,'OutputView', cb_ref);
            data.joint4(:,:,i) = imwarp(data.joint4(:,:,i), tform04,'OutputView', cb_ref);
        else
            data.joint0_bw(:,:,i) = imwarp(data.joint0_bw(:,:,i), tform00,'OutputView', cb_ref);
            data.joint0(:,:,i) = imwarp(data.joint0(:,:,i), tform00,'OutputView', cb_ref);
        end
    end

    % Finding optimal transformation for TOP registration on each joint
    if data.full_volume  % If the volume is the whole hand         
        [tform11,~] = register(double(data.joint1_bw(:,:,data.base_slice_2+1)), double(data.joint1_bw(:,:,data.base_slice_2)),1,params);
        [tform12,~] = register(double(data.joint2_bw(:,:,data.base_slice_2+1)), double(data.joint2_bw(:,:,data.base_slice_2)),1,params);
        [tform13,~] = register(double(data.joint3_bw(:,:,data.base_slice_2+1)), double(data.joint3_bw(:,:,data.base_slice_2)),1,params);
        [tform14,~] = register(double(data.joint4_bw(:,:,data.base_slice_2+1)), double(data.joint4_bw(:,:,data.base_slice_2)),1,params);
    else
        [tform10,~] = register(double(data.joint0_bw(:,:,data.base_slice_2+1)), double(data.joint0_bw(:,:,data.base_slice_2)),1,params);
    end
    
    % Applying TOP transformation on each joint
    for i = data.base_slice_2+1:data.n_slices        
        if data.full_volume  % If the volume is the whole hand     
            data.joint1_bw(:,:,i) = imwarp(data.joint1_bw(:,:,i), tform11,'OutputView', cb_ref);
            data.joint1(:,:,i) = imwarp(data.joint1(:,:,i), tform11,'OutputView', cb_ref);
            data.joint2_bw(:,:,i) = imwarp(data.joint2_bw(:,:,i), tform12,'OutputView', cb_ref);
            data.joint2(:,:,i) = imwarp(data.joint2(:,:,i), tform12,'OutputView', cb_ref);
            data.joint3_bw(:,:,i) = imwarp(data.joint3_bw(:,:,i), tform13,'OutputView', cb_ref);
            data.joint3(:,:,i) = imwarp(data.joint3(:,:,i), tform13,'OutputView', cb_ref);
            data.joint4_bw(:,:,i) = imwarp(data.joint4_bw(:,:,i), tform14,'OutputView', cb_ref);
            data.joint4(:,:,i) = imwarp(data.joint4(:,:,i), tform14,'OutputView', cb_ref);
        else
            data.joint0_bw(:,:,i) = imwarp(data.joint0_bw(:,:,i), tform10,'OutputView', cb_ref);
            data.joint0(:,:,i) = imwarp(data.joint0(:,:,i), tform10,'OutputView', cb_ref);    
        end
    end

    % Recombine joints into a full volume
    if data.full_volume  % If the volume is the whole hand  
        data.im_bw = logical(data.joint1_bw+data.joint2_bw+data.joint3_bw+data.joint4_bw);
        data.im_seg = data.joint1+data.joint2+data.joint3+data.joint4;
    else
        data.im_bw = logical(data.joint0_bw);
        data.im_seg = data.joint0;
    end

    % Save metrics needed to de-registrate the final erosions
    data = save_bb_masks(data);
    if data.full_volume  % If the volume is the whole hand         
        data.inv_tform01 = invert(tform01);
        data.inv_tform02 = invert(tform02);
        data.inv_tform03 = invert(tform03);
        data.inv_tform04 = invert(tform04);
        data.inv_tform11 = invert(tform11);
        data.inv_tform12 = invert(tform12);
        data.inv_tform13 = invert(tform13);
        data.inv_tform14 = invert(tform14);
    else
        data.inv_tform00 = invert(tform00);
        data.inv_tform10 = invert(tform10);
    end
    
%     opengl hardware
%     figure(); volshow(data.joint0_bbs);
%     figure(); volshow(data.joint0_bbs+uint16(data.joint0_bw));
    
    fprintf("%s Registration complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

