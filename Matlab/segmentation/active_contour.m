function data = active_contour(data)
%ACTIVE_CONTOUR Refines the border of the bone surface with active contour. (AC)
    time = tic;

    % Init waitbar
    if data.wb  % if waitbars is initialized
        n_monitors = size(get(0,'MonitorPositions'), 1);
        if n_monitors == 1
            wb = waitbar(0,'Segmentation - Active Contour');
        else
            wb = waitbar(0,'Segmentation - Active Contour', 'Position', [-800,500,270,50]);
        end
    end
    
    % Init variables
    rad = 18;
    alpha = 0.08;
    num_it = 20;
    epsilon = 1;

    % Perform AC on every slice
    par_im_bw = data.im_bw;
    if data.parallel
        parfor i = 1:data.n_slices
            par_im_bw(:,:,i) = local_AC_UM(im2double(data.im_adj(:,:,i)),im2double(data.im_bw(:,:,i)),rad,alpha,num_it,epsilon);
        end
    else
        for i = 1:data.n_slices
            par_im_bw(:,:,i) = local_AC_UM(im2double(data.im_adj(:,:,i)),im2double(data.im_bw(:,:,i)),rad,alpha,num_it,epsilon);
            if data.wb  % if waitbars is initialized
                waitbar(i/data.n_slices)  % update waitbar
            end
        end
    end
    data.im_bw = par_im_bw;
    
    if data.wb  % if waitbars is initialized
        close(wb)  % close waitbar
    end
    fprintf("%s Segmenting - Active contour complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
end

