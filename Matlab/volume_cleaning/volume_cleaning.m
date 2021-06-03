function data = volume_cleaning(data)
%VOLUME_CLEANING Not described in FM thesis. Is speculated to remove
%various artifacts.
    time = tic;  % start timer
    
    % Run scripts
    data.im_clean = joint_splitter(data.im_bw);
    data.im_clean = element_removal(data.im_clean, data.full_volume);
    
    fprintf("%s Volume cleaning complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
    
end