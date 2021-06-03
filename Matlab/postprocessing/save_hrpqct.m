function data = save_hrpqct(data)
%SAVE_HRPQCT Saves data and truth, and changes filename to the correct for
%the generated label map
    time = tic;  % start timer

    % Change filename for output
    data.output_filename = data.input_filename(3:end);

    % Ensure output folder exists
    if exist(data.output_path, 'dir') ~= 7  
        mkdir(data.output_path);
    end
    
    % Get input paths
    data_in    = [data.input_path, data.input_filename, '.nii.gz'];
    truth_in    = [data.input_path, '2_', data.output_filename, '_label.nii.gz'];
    
    % Get output paths
    data_out   = [data.output_path, data.output_filename, '_data.nii.gz'];
    truth_out   = [data.output_path, data.output_filename, '_truth.nii.gz'];
    
    % Copy files
    copyfile(data_in, data_out)
    copyfile(truth_in, truth_out)

    fprintf("%s save_hrpqct complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));
end