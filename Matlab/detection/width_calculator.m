function [len,pixel1,pixel2,slice] = width_calculator(erosion, imageSet)
%WIDTH_CALCULATOR Summary of this function goes here. (widthCalculator2)
    
    rp = regionprops3(erosion);
    center = round(rp.Centroid(3));
    
    full = imageSet(:,:,center) + erosion(:,:,center);
    full = imclose(full, strel('disk',3));
    inner = imerode(full, strel('disk', 1));
    outer = imabsdiff(logical(full),logical(inner));
    
    side = erosion(:,:,center).*outer;

    dist = zeros(1,8*8);
    pos1 = zeros(2,8*8);
    pos2 = zeros(2,8*8);
    
    rp = regionprops(side,'Area', 'Extrema');
    rpArea = [rp.Area];
    [~, index] = max(rpArea);
    
    if isempty(index) % if the erosion does not overlap with 'outer'
        disp('width_calculator.m:ERROR:EROSION NOT FOUND');
        len = 0;
        pixel1 = [0,0];
        pixel2 = [0,0];
        slice = center;
    else
        extrema = rp(index).Extrema;
        for j = 1:8
            for k = 1:8
                x1 = extrema(j,1);
                x2 = extrema(k,1);
                y1 = extrema(j,2); 
                y2 = extrema(k,2);
                pos1(1,j*8+k) = x1;
                pos1(2,j*8+k) = y1;
                pos2(1,j*8+k) = x2;
                pos2(2,j*8+k) = y2;
                dist(1,j*8+k) = sqrt((x1-x2)^2+(y1-y2)^2);     
            end
        end
        [len, max_index] = max(dist);
        pixel1 = pos1(:,max_index);
        pixel2 = pos2(:,max_index);
        slice = center;
    end
end

