function [tform, im] = register(moving, fixed, imTrue, params)
%REGISTER Performs optimal registration by reiteration. Returns general
%transform to be used for all other slices.

    [optimizer, metric] = imregconfig('monomodal');
    
    optimizer.GradientMagnitudeTolerance = params.GradientMagnitudeTolerance;
    optimizer.MinimumStepLength = params.MinimumStepLength;
    optimizer.MaximumStepLength = params.MaximumStepLength;
    optimizer.MaximumIterations = params.MaximumIterations;
    optimizer.RelaxationFactor = params.RelaxationFactor;
    
    tform = imregtform(moving, fixed, 'rigid', optimizer, metric);
    
    if imTrue == 1
        cb_ref = imref2d(size(moving));
        [im,~] = imwarp(moving,tform,'OutputView', cb_ref);
    else
        im = 0;
    end
    
end

