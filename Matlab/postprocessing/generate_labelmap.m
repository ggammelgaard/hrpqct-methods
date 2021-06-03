function data = generate_labelmap(data)
%GENERATE_LABELMAP Creates labelmap and save it to path in NIFTI format
    time = tic;  % start timer

    % Erosion identifier class
    class = 3;
    
    % Create labelmap
    data.labelmap = int8(logical(data.erosions))*class;

    % 'De-register' the labelmap, to fit GT
    data = deregister(data);
    
    % Load and modify info from input volume
    info = niftiinfo([data.input_path, data.input_filename, '.nii.gz']);
    info.Datatype = 'int8';
    
    % Ensure output folder exists
    if exist(data.output_path, 'dir') ~= 7  
        mkdir(data.output_path);
    end
    
    % Save labelmap
    output_path = ([data.output_path, data.output_filename, '_prediction']);
    niftiwrite(data.labelmap, output_path, info, 'Compressed', true) 
    
    % Save im_bw (for debugging purposes)
    if 1
        im_bw = data.im_bw;
        save([data.output_path, data.output_filename, '_im_bw.mat'],'im_bw')
    end
    
    fprintf("%s Generate labelmap complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
end