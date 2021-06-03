function data = section_detector(data)
%SECTION_DETECTOR Find slices where the three sections are combined

    % Init array
    mse_array1 = zeros(15, 1);

    % Find first connection slice (originally 110)
    for i = 1:15
        mse_array1(i) = immse(data.im_adj(:,:,i+100), data.im_adj(:,:,i+101));
    end

    % Find largest mse
    [~, idx] = max(mse_array1);
    
    % Save to data
    data.base_slice_1 = 100 + idx;
    data.base_slice_2 = data.base_slice_1 + 110;  % second connection is always 110 slices above

end
