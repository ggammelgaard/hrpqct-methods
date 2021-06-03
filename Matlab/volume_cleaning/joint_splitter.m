function output_tensor = joint_splitter(input_tensor)
%JOINTSPLITTER Ensures that each join is split in two volumes. (jointSplitterv2)

    % Get n_slices
    [~,~,n_slices] = size(input_tensor);
    % Init output_tensor
    output_tensor = input_tensor;

    % Find connected components and get their bounding boxes
    cc = bwconncomp(input_tensor);
    rp = regionprops3(input_tensor);   
 
    
    for i = 1:size(rp,1)    
        % If bounding box goes bottom to top the joint has not been split
        % and small connection must exist between the two joints
        if floor(rp.BoundingBox(i,3)) == 0 && rp.BoundingBox(i,6) == n_slices
            % Create volume with the faulty component
            mcp = false(size(input_tensor));
            mcp(cc.PixelIdxList{i}) = 1;

            % Find center and area for every slice of the component
            x_center = zeros(1,n_slices);
            y_center = zeros(1,n_slices);
            area = zeros(1,n_slices);
            for j = 1:n_slices
                rp2 = regionprops(mcp(:,:,j));
                [~,elem] = max([rp2.Area]);
                x_center(1,j) = rp2(elem).Centroid(1);
                y_center(1,j) = rp2(elem).Centroid(2);
                area(1,j) = rp2(elem).Area;
            end

            % Find differences between each consecutive slice of component
            x_diff = diff(x_center);
            y_diff = diff(y_center);
            area_diff = diff(area);
            % Find slices with min and max difference
            [~,x_slice_max] = max(x_diff);
            [~,x_slice_min] = min(x_diff);
            [~,y_slice_max] = max(y_diff);
            [~,y_slice_min] = min(y_diff);
            [~,area_slice_max] = max(area_diff);
            [~,area_slice_min] = min(area_diff);

            % See if the x slices are somewhere in the y slice set
            mem = ismember([x_slice_max,x_slice_min],[y_slice_min,y_slice_max]);
            if sum(mem) == 1
                if mem(1,1) == 1  % Finds which slice that was matching
                    slice = x_slice_max;
                else
                    slice = x_slice_min;
                end
            else  % If no overlap was found between x and y
                % See if area slices are overlap with any x or y slices
                mem = ismember([area_slice_min,area_slice_max],[y_slice_min,y_slice_max,x_slice_min,x_slice_max]);
                if mem(1,1) == 1  % Finds which slice that was matching
                    slice = area_slice_min;
                elseif mem(1,2) == 1
                    slice = area_slice_max;
                else  % If no matches was found just find the smallest difference between any two
                    slices_sorted = sort([area_slice_min,area_slice_max,y_slice_min,y_slice_max,x_slice_min,x_slice_max]);
                    slices_inc = diff(slices_sorted);
                    [~,pos] = min(slices_inc);
                    slice = slices_sorted(pos);
                end
            end

            if 0
                % First try: WATERSHED SOLUTION:
                % https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/
                slice_width = 10;  % Can be changed
                slice_range = (slice-slice_width:slice+slice_width);
                slice_range(slice_range <= 0) = [];  % Ensure range is within volume
                slice_range(slice_range > n_slices) = [];  % Ensure range is within volume

                split_slices = mcp(:,:,slice_range);
                % 3D
                D = bwdist(~split_slices);
                mask = ~imextendedmin(D,6);
                D2 = imimposemin(D,mask);
                Ld2 = watershed(D2);
                split_slices(Ld2 == 0) = 0;
                mcp(:,:,slice_range) = split_slices;

                            % Check if it worked
                tmp_cc = bwconncomp(mcp);
                if tmp_cc.NumObjects >= 2 
                    % Update mcp
                    watershed_flag = true;    
                else  % If not
                    % ORIG SOLUTION:
                    % Set all values to zero in the slice, as well as above and below
                    mcp(cc.PixelIdxList{i}) = 1;  % load mcp again
                    delete_range = (slice-3:slice+1);
                    delete_range(delete_range <= 0) = [];  % Ensure range is within volume
                    delete_range(delete_range > n_slices) = [];  % Ensure range is within volume
                    mcp(:,:,delete_range) = 0;
                    watershed_flag = false;
                end
            end
            
            % ORIG SOLUTION:
            % Set all values to zero in the slice, as well as above and below
            mcp(cc.PixelIdxList{i}) = 1;  % load mcp again
            delete_range = (slice-1:slice+1);
            delete_range(delete_range <= 0) = [];  % Ensure range is within volume
            delete_range(delete_range > n_slices) = [];  % Ensure range is within volume
            mcp(:,:,delete_range) = 0;
            watershed_flag = false;

            % Update the output tensor with the mcp
            output_tensor(cc.PixelIdxList{i}) = 0;  % Remove the offending component
            output_tensor = output_tensor + mcp;  % Add the fixed component
            output_tensor = logical(output_tensor);  % Ensure it is still logical

            % Print
            %sfprintf("%s JOINT SPLIT. Used watershed method: %s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), string(watershed_flag));
            
        end
    end    
end