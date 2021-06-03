function final_erosions = remove_outer_objects(erosions,input_tensor)
%REMOVE_OUTER_OBJECTS Remove erosions found on the outside of the bone surface (removeOuterObjects)

    eor = bwareaopen(erosions, 100);
    cc = bwconncomp(eor);

    final_erosions = false(size(eor));

    for i = 1:cc.NumObjects %-1
        ero = false(size(eor));
        ero(cc.PixelIdxList{i}) = 1;
       
        [~,pixel1,pixel2,slice] = width_calculator(ero, input_tensor);
        [~,point] = depth_calculator(ero, pixel1, pixel2, slice);

        combined = logical(input_tensor + ero);
        rpSlice = regionprops(combined(:,:,slice));

        for j =  1:size(rpSlice, 1)
            if point(1) > rpSlice(j).BoundingBox(1)...
                && point(1) < rpSlice(j).BoundingBox(1) + rpSlice(j).BoundingBox(3)...
                && point(2) > rpSlice(j).BoundingBox(2)...
                && point(2) < rpSlice(j).BoundingBox(2) + rpSlice(j).BoundingBox(4)

                lenght1 = sqrt((point(1)-rpSlice(j).Centroid(1))^2+(point(2)-rpSlice(j).Centroid(2))^2);   
                lenght2 = sqrt((pixel1(1)-rpSlice(j).Centroid(1))^2+(pixel1(2)-rpSlice(j).Centroid(2))^2); 
                lenght3 = sqrt((pixel2(1)-rpSlice(j).Centroid(1))^2+(pixel2(2)-rpSlice(j).Centroid(2))^2); 

                if max(lenght2,lenght3) > lenght1
                   final_erosions = final_erosions + ero;
                end
            end
        end
    end
end

