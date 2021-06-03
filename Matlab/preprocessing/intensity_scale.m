function data = intensity_scale( data )
%PREPROCESS Create new tensor with adjusted intensity values. (imAdjust)
    time = tic;

    % initialize variables
    data.im_adj = uint16(zeros(size(data.im_data)));
    set_min = zeros(1, data.n_slices);  
    set_max = zeros(1, data.n_slices);  

    % Convert data from int16 to uint16
    for i = 1:data.n_slices  
        data.im_adj(:,:,i) = im2uint16(data.im_data(:,:,i));  
    end

    % Find smallest value for each slice
    for i = 1:data.n_slices  
        set_min(1,i) = min(min(data.im_adj(:,:,i)));
        set_max(1,i) = max(max(data.im_adj(:,:,i)));
    end

    % Find global intensity range from median of all slices
    low_in = double(median(set_min))/2^16;
    high_in = double(median(set_max))/2^16;

    % Scale intensity values to use the whole value range
    for k=1:data.n_slices  
        data.im_adj(:,:,k) = imadjust(data.im_adj(:,:,k), [low_in, high_in]);
    end
    
    fprintf("%s Preprocessing - Intensity scaling complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

