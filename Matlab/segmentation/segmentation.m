function data = segmentation(data)
%SEGMENTATION Bone segmentation.
    time = tic;
    
    data = thresholder(data);
    
%     opengl hardware
%     figure(); volshow(data.im_bw);
    
    data = morph_ops(data);
    
%     opengl hardware
%     figure(); volshow(data.im_bw);
    
    data = active_contour(data);
    
    data.im_seg = uint16(data.im_adj);
    data.im_seg(~data.im_bw) = 0;
  
%     opengl hardware
%     figure(); volshow(data.im_bw);
    
    fprintf("%s Segmenting complete. Duration: %3.2f s\n", datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(time));

end

