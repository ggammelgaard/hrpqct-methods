function [max_depth,pixel] = depth_calculator(erosion, pixel1, pixel2, slice)
%DEPTH_CALCULATOR Summary of this function goes here. (depthCalculator)
erosion_border = bwmorph(erosion(:,:,slice), 'remove');

[M,N] = size(erosion_border);
x1 = pixel1(1);
y1 = pixel1(2);
x2 = pixel2(1);
y2 = pixel2(2);

n_pixels = sum(sum(erosion_border));
depths = zeros(1,n_pixels);
xVals = zeros(1,n_pixels);
yVals = zeros(1,n_pixels);
it = 1;

for i = 1:M
    for j = 1:N
        if erosion_border(i,j) == 1
            depths(1,it) = abs( (y2-y1)*j-(x2-x1)*i+x2*y1-y2*x1)/sqrt((y2-y1)^2+(x2-x1)^2);
            xVals(1,it) = j;
            yVals(1,it) = i;
            it = it +1;       
        end      
    end
end

[max_depth, max_index] = max(depths);
pixel = [xVals(1,max_index), yVals(1,max_index)];

end

