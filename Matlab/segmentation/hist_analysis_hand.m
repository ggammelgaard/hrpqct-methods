function data = hist_analysis_hand(data)
%HISTOGRAM_ANALYSIS_HAND Fits gaussian curves to image histogram. (histAnalysis2)

    hist = histcounts(data.im_adj, 2^16);  % Generate histogram
    x = 1:2^16;  % Initialize x-axis

    % Only keep bins that are not empty
    x = x(hist > 0);
    hist = hist(hist > 0); 

    % Perform curve fitting
    f = fit(x(:),hist(:),'gauss2');

    % Assign variables
    center = [f.b1, f.b2];
    width  = [f.c1, f.c2];
    height = [f.a1, f.a2];
    [~, index1] = min(center);
    [~, index2] = max(center);
    data.tissue.center   = center(index1);
    data.tissue.width    = width(index1);
    data.tissue.height   = height(index1);
    data.bone.center  = center(index2);
    data.bone.width   = width(index2);
    data.bone.height  = height(index2);

    if 0
        tissue = data.tissue.height*exp(-((x-data.tissue.center)/data.tissue.width).^2);
        bone = data.bone.height*exp(-((x-data.bone.center)/data.bone.width).^2);
        combined = tissue+bone;

        opengl software; figure();
        plot(x, hist, 'linewidth', 2); hold on
        plot(x, tissue, 'linewidth', 2); hold on
        plot(x, bone, 'linewidth', 2); hold on 
        plot(x, combined, 'linewidth', 2)
        legend('Orig', 'Tissue', 'Bone', 'Combined')

        ylim([0 8000]);

        % add vline at thresholding value
        thresh = (data.bone.center+(2.5*data.bone.width));
        yL = get(gca,'YLim');
        line([thresh thresh],yL,'Color','r');
    end
end

