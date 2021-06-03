function data = hist_analysis_finger(data)
%HISTOGRAM_ANALYSIS_FINGER Fits gaussian curves to image histogram. (histAnalysis2)

    hist = histcounts(data.im_adj, 2^16);  % Generate histogram
    x = 1:2^16;  % Initialize x-axis

    % Only keep bins that are not empty
    x = x(hist > 0);
    hist = hist(hist > 0); 

    % Perform curve fitting
    bkgrnd_zero = find(hist == max(hist(:))); % Find the indice for the extreme background pixel
    data.hist_zero = x(bkgrnd_zero);  % Used in Kang paper
    % We also exclude beginning and end, as they can have a high number of values
    f = fit(x(:),hist(:),'gauss2', 'Exclude', [1 bkgrnd_zero size(x,2)], 'Lower', [0,0,0,0,0,0]);

    % Assign variables
    center = [f.b1, f.b2];
    width  = [f.c1, f.c2];
    height = [f.a1, f.a2];
    [~, index1] = min(center); 
    [~, index2] = max(center); 
    data.tissue.center      = center(index1);
    data.tissue.width       = width(index1);
    data.tissue.height      = height(index1);
    data.bone.center        = center(index2);
    data.bone.width         = width(index2);
    data.bone.height        = height(index2);

    % Find Kang's LT and HT
    % Original intersection of curves method
    a = 1/(data.bone.width^2) - 1/(data.tissue.width^2);
    b = 2*(data.tissue.center/(data.tissue.width^2) - data.bone.center/(data.bone.width^2));
    c = data.bone.center^2/(data.bone.width^2) - data.tissue.center^2/(data.tissue.width^2) + log(data.tissue.height) - log(data.bone.height);
    d = b^2 - 4 * a * c;
    x1 = (-b + sqrt(d))/(2*a);
    x2 = (-b - sqrt(d))/(2*a);
    hist_intersect = max(x1,x2);

    data.low_thresh = hist_intersect + data.kang_beta*(2^16)^data.kang_gamma/(data.tissue.width)^data.kang_gamma;%1.5*(data.tissue.center-data.hist_zero);
%     if isreal(data.low_thresh) == false  % To avoid imaginary numbers errors
%         data.low_thresh = real(data.low_thresh);
%     end
    data.high_thresh = data.low_thresh + data.kang_tp; % 10000 % 4800

    if 0
        tissue = data.tissue.height*exp(-((x-data.tissue.center)/data.tissue.width).^2);
        bone = data.bone.height*exp(-((x-data.bone.center)/data.bone.width).^2);
        combined = tissue+bone;

        set(0,'DefaultTextFontname', 'CMU Serif')
        set(0,'DefaultAxesFontName', 'CMU Serif')

        opengl software; 
        fig = figure();
        h = area(x, hist, 'EdgeColor', '#C0C0C0', 'FaceColor', '#C0C0C0'); hold on
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
%         h = line([data.hist_zero data.hist_zero],[0 9000],'LineStyle',':','Color','k', 'linewidth', 1.5);
%         h.Annotation.LegendInformation.IconDisplayStyle = 'off';
%         text(data.hist_zero-2600 ,-200,'0 HU','FontSize',10, 'FontWeight','bold')
        plot(x, tissue, 'linewidth', 2, 'color', '#e41a1c'); hold on
        plot(x, bone, '--', 'linewidth', 2, 'color', '#377eb8'); hold on 
        plot(x, combined, 'linewidth', 3, 'color', '#984ea3')
        % Cross section markers
        [~,ix] = min(abs(x-x1));
        plot(x1,tissue(ix),'o', 'color','k', 'MarkerFaceColor', 'w')
        text(x1+1800 ,tissue(ix),'p_1','FontSize',10, 'FontWeight','bold')
        [~,ix] = min(abs(x-x2));
        plot(x2,tissue(ix),'o', 'color','k','MarkerFaceColor', 'w')
        text(x2+1000 ,tissue(ix)-100,'p_2','FontSize',10, 'FontWeight','bold')
        % LT HT lines
%         yL = [0 6000];
%         line([data.low_thresh data.low_thresh],yL,'Color','#4C4C4C', 'linewidth', 2);
%         line([data.high_thresh data.high_thresh],yL,'Color','#4C4C4C', 'linewidth', 2);
%         text(data.low_thresh-1700 ,yL(2)+300,'LT','FontSize',12)
%         text(data.high_thresh-1700 ,yL(2)+300,'HT','FontSize',12)
        % Visual
        xlim([-1000 2^16+1000]);
        ylim([0 10000]);
        xlabel('Voxel intensity value')
        ylabel('Number of voxels')
        legend('Soft tissue', 'Bone', 'y(x)');
        set(gca,'FontSize',12)

        if 1
            set(0,'DefaultTextFontname', 'CMU Serif')
            set(0,'DefaultAxesFontName', 'CMU Serif')

            opengl software; figure();
            area(x, hist, 'EdgeColor', '#C0C0C0', 'FaceColor', '#C0C0C0'); hold on
            xlim([-1000 2^16+1000]);
            ylim([0 10000]);
            yL = [0 7200];
            line([data.low_thresh data.low_thresh],yL,'Color','#4C4C4C', 'linewidth', 2);
            line([data.high_thresh data.high_thresh],yL,'Color','#4C4C4C', 'linewidth', 2);
            text(data.low_thresh-1700 ,yL(2)+300,'LT','FontSize',12)
            text(data.high_thresh-1700 ,yL(2)+300,'HT','FontSize',12)
            xlabel('Voxel intensity value')
            ylabel('Number of voxels')
            set(gca,'FontSize',12)

        end
    end
end

