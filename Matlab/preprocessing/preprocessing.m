function data = preprocessing(data)
%PREPROCESSING Loads in data and performs intensity scaling.
    time = tic;  % start timer

    % Loads in the dataset
    data = load_dataset(data);
    
    % Performs image adjustment for the image colors
    data = intensity_scale(data);  
    
    % Init parpool
    if data.parallel
        start_parpool();
    end
    
    % Find slices where the three sections are combined
    data = section_detector(data);
    
    % Print
    fprintf("%s Preprocessing complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

