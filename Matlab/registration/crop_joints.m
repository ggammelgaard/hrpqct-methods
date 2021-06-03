function data = crop_joints(data)
%CROP_JOINTS Used for hand volumes. Creates independent volumes for each joint (cropJoints)
    time = tic;

    % Init data variables
    data.joint1_bw = logical(false(size(data.im_bw)));
    data.joint2_bw = data.joint1_bw;
    data.joint3_bw = data.joint1_bw;
    data.joint4_bw = data.joint1_bw;
    data.joint1 = uint16(zeros(size(data.im_bw)));
    data.joint2 = data.joint1;
    data.joint3 = data.joint1;
    data.joint4 = data.joint1;
    % Init local variables
    x = zeros(1,4);
    elems = 0;

    % Loop through the first 10 slices (?)
    for i = 1:10
        [label_bw, num_elems] = bwlabel(data.im_bw(:,:,i)); 
        if num_elems == 4
            elems = elems + 1;
            rp = regionprops(label_bw, 'Centroid');
            for j = 1:4
                x(1,j) = x(1,j) + rp(j).Centroid(1);
            end
        end
    end
    x = x/elems;
    regions = [0, ((x(1)+x(2))/2), ((x(2)+x(3))/2), ((x(3)+x(4))/2), size(data.im_bw(:,:,i),2)];

    % Loop through all slices
    for k = 1:data.n_slices        
        [label_bw, num_elems] = bwlabel(data.im_bw(:,:,k)); 

        rp = regionprops(label_bw, 'Centroid');
        cc = bwconncomp(data.im_bw(:,:,k), 8);

        for l = 1:num_elems
            tmp_elem = false(size(data.im_bw(:,:,k)));
            tmp_elem(cc.PixelIdxList{l}) = true;
            if rp(l).Centroid(1) > regions(1) && rp(l).Centroid(1) < regions(2)
                data.joint1_bw(:,:,k) = data.joint1_bw(:,:,k) + double(tmp_elem(:,:,1));
            elseif rp(l).Centroid(1) > regions(2) && rp(l).Centroid(1) < regions(3) 
                data.joint2_bw(:,:,k) = data.joint2_bw(:,:,k) + double(tmp_elem(:,:,1));
            elseif rp(l).Centroid(1) > regions(3) && rp(l).Centroid(1) < regions(4) 
                data.joint3_bw(:,:,k) = data.joint3_bw(:,:,k) + double(tmp_elem(:,:,1));
            elseif rp(l).Centroid(1) > regions(4) && rp(l).Centroid(1) < regions(5) 
                data.joint4_bw(:,:,k) = data.joint4_bw(:,:,k) + double(tmp_elem(:,:,1));
            end
        end

        tmp_joint1 = uint16(data.im_adj(:,:,k));
        tmp_joint1(~data.joint1_bw(:,:,k)) = 0;
        data.joint1(:,:,k) = tmp_joint1;

        tmp_joint2 = uint16(data.im_adj(:,:,k));
        tmp_joint2(~data.joint2_bw(:,:,k)) = 0;
        data.joint2(:,:,k) = tmp_joint2;

        tmp_joint3 = uint16(data.im_adj(:,:,k));
        tmp_joint3(~data.joint3_bw(:,:,k)) = 0;
        data.joint3(:,:,k) = tmp_joint3;

        tmp_joint4 = uint16(data.im_adj(:,:,k));
        tmp_joint4(~data.joint4_bw(:,:,k)) = 0;
        data.joint4(:,:,k) = tmp_joint4;
    end
    
    fprintf("Cropping joints complete. Duration: %3.2f s\n", toc(time));

end

