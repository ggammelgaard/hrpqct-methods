function data = load_dataset(data)
%LOAD Load nii.gz file. (loadDataset)
    time = tic;  % start timer
    
    data_path    = [data.input_path, data.input_filename, '.nii.gz'];
    data.im_data = int16(niftiread(data_path));  % load tensor
    data.im_info = niftiinfo(data_path);  % load meta info
    
    % Info used by the rest of the script
    data.width = size(data.im_data,1);
    data.depth = size(data.im_data,2);
    data.n_slices = size(data.im_data,3);  % number of slices in tensor
        
    % Print
    fprintf("%s Preprocessing - Load dataset complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
    
end

