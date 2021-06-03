function erosions_refined = find_connected_objects(erosions,image_set, n_slices)
%FIND_CONNECTED_OBJECTS Check if detected erosions are part of larger erosion (findConnectedObjects)

    full = imclose(image_set, strel('cube',50));
    erosions_full = imabsdiff(full, image_set);
    erosions_full = bwmorph3(erosions_full, 'majority');
    erosions_full = imopen(erosions_full, strel('cube', 10));
    erosions_full = bwareaopen(erosions_full, 200);
    full_refined  = logical(image_set + erosions_full);

    for i = 1:n_slices
        full_refined(:,:,i) = imfill(full_refined(:,:,i), 'holes');
    end

    erosion_refine = imabsdiff(full_refined, image_set);
    cc = bwconncomp(erosion_refine);
    rp = regionprops3(logical(erosions));
    final_object = false(size(erosions));
    for i = 1:cc.NumObjects  % Loop through all erosions
        object = false(size(erosions));
        object(cc.PixelIdxList{i}) = 1;

        n_erosions = 0;
        for j = 1:size(rp,1)

            if object(round(rp.Centroid(j,2)),round(rp.Centroid(j,1)),round(rp.Centroid(j,3))) == 1
                n_erosions = n_erosions +1;
            end
        end

        if n_erosions > 1
            final_object = final_object + object;
        end
    end

    erosions_refined = logical(erosions + final_object);

end

