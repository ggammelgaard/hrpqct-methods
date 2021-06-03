function data = kang_adaptive(data)
%KANG_ADAPTIVE Implements the adaptive thresholding method described by
%Kang et. al in "A new accurate and precise 3-D segmentation method for
%skeletal structures in volumetric CT data" from 2003

    % All voxel intensities above high tresh are here to stay
    data.im_bw = imbinarize(data.im_adj, data.high_thresh/2^16);

%     opengl hardware
%     figure(); volshow(data.im_bw);

    % Finding voxels between low_tresh and high_tresh
    im_mid = imbinarize(data.im_adj, data.low_thresh/2^16);  % remove voxels below low tresh
    im_mid = logical(im_mid - data.im_bw);  % subtract im_high voxels

    % Adaptive thresholding of im_mid
    % Prepare vars for parallel execution
    width = data.width; depth = data.depth; n_slices = data.n_slices;
    im_adj = data.im_adj; tmp_im_bw = data.im_bw;
    alpha = data.kang_alpha;
    if data.parallel
        parfor x = 1:width
            tmp_im_bw(x,:,:) = adaptive_loop(x, width, depth, n_slices, im_mid, im_adj, tmp_im_bw(x,:,:), alpha);
        end
    else
        for x = 1:width
            tmp_im_bw(x,:,:) = adaptive_loop(x, width, depth, n_slices, im_mid, im_adj, tmp_im_bw(x,:,:), alpha);
        end
    end
        
    data.im_bw = tmp_im_bw;  % Update real variable
end

function tmp_im_bw_x = adaptive_loop(x, width, depth, n_slices, im_mid, im_adj, tmp_im_bw_x, alpha)
        for y = 1:depth
            for z = 1:n_slices
                if im_mid(x,y,z) == true  % if pixel should be evaluated
                    % Find 3x3x3 neighbourhood (26 voxel)
                    x_range = min(max(x-1:x+1,1),width);
                    y_range = min(max(y-1:y+1,1),depth);
                    z_range = min(max(z-1:z+1,1),n_slices);

                    neighbours = im_adj(x_range, y_range, z_range);
                    neighbours = neighbours(:);
                    neighbours(14) = [];  % Remove the middle voxel

                    mu = mean(double(neighbours));
                    sigma = std(double(neighbours));

                    % Equation from article. If above, the voxel is bone
                    if im_adj(x,y,z) >= (mu-alpha*sigma)
                        tmp_im_bw_x(1,y,z) = 1;  % update im_bw
                    end
                end
            end
        end
end