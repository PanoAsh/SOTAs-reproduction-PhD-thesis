clc,clear;
folder = '/home/yzhang1/MatlabProjects/erp';

imlist = dir(folder);

output_size = 512;
vfov = 90;
headmove_h = 0:30:60;
headmove_v = 0:30:60;


Imgs = dir(fullfile(folder,'*.png'));

for i = 1:length(Imgs)
   
     
    % read image
    img = imread(fullfile(folder,Imgs(i).name));
    imw = size(img, 2);
    iml = size(img, 1);
    
  
    
    for hh = 1:length(headmove_h)
        offset = round(headmove_h(hh)/360*imw);
      
        im_turned = [img(:, imw-offset+1:imw, :) img(:, 1:imw-offset, :)];
      
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, output_size, vfov, headmove_v(hv));
           
            for f=1:6
                debug=cell2mat(out(f))
                filename = ['Img_' num2str(i) '_' num2str(hh) '_' num2str(hv) '_' num2str(f) '.png']
                imwrite(debug, filename);
               
            end
       end     
    end
end