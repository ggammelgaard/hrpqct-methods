function data = thresholder(data)
%THRESHOLDER Finds global threshold and applies it to data.im_adj (Thresholder)
    time = tic;

    if data.full_volume   
        % Find gaussian curves in histogram
        data = hist_analysis_hand(data);
        % Perform thresholding (globalThresholder)
        thresh = (data.bone.center+(2.5*data.bone.width))/2^16;
        data.im_bw = imbinarize(data.im_adj, thresh);
    else
        % Find gaussian curves in histogram
        data = hist_analysis_finger(data);
        % Perform thresholding
        data = kang_adaptive(data);
    end
    
    % Print status
    fprintf("%s Segmenting - Thresholding complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

