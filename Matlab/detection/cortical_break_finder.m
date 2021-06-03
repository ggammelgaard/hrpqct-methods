function [cortical_breaks,cortical] = cortical_break_finder(full, breaks, size)
%CORTICAL_BREAK_FINDER Summary of this function goes here. (CorticalBreakFinder)

    % Cortical border
    cortical = imerode(full, strel('sphere',4));
    cortical = logical(imabsdiff(full,cortical));

    % Find breaks that goes all the way through the cortical border
    cortical_breaks = imsubtract(breaks,cortical);
    cortical_breaks(cortical_breaks == -1) = 0;

    % Removes voxels that does not have any neighbours in their 3x3x3 area
    cortical_breaks = bwmorph3(cortical_breaks,'clean');
    % Removes voxels that does not have more than 14 neighbours in their 3x3x3 area
    cortical_breaks = bwmorph3(cortical_breaks,'majority');
    
    % Remove components that are smaller than 'size' pixels (100)
    cortical_breaks = bwareaopen(cortical_breaks, size);
    
end

