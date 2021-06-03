function output_tensor = element_removal(input_tensor, full_volume_flag)
%UNTITLED Removes elements that are not the two bones for each finger. (elementRemoval)
    
    % Find connected components and get their volumes
    cc = bwconncomp(input_tensor);
    rp = regionprops3(input_tensor);
    
    % Check if there is more components than N_fingers * 2
    if full_volume_flag  % If the volume is the whole hand
        n_components = 8;  % 2 bones * 4 fingers
    else
        n_components = 2;  % 2 bones * 1 finger
    end
    
    % If there is too many elements
    if size(rp,1) > n_components
        % Get volumes of all components
        volume_vals = zeros(1,size(rp,1));
        for i = 1:size(rp,1)
            volume_vals(1,i) = rp.Volume(i);
        end
        % Sort all components based on volume
        [~, max_ids] = sort(volume_vals, 'descend'); 

        % Insert the n biggest components in a new tensor
        output_tensor = false(size(input_tensor));
        for i = 1:n_components
            component = false(size(input_tensor));
            component(cc.PixelIdxList{max_ids(i)}) = 1;
            output_tensor = output_tensor + component;
        end
        
        % Ensure that the output is logical
        output_tensor = logical(output_tensor);
    
    else
        output_tensor = input_tensor; 
    end
end

