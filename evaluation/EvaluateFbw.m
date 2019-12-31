close all;clear all;clc;

%select the Ground truth dataset path
gtPath = '/Users/zhangyi/Projects/MatlabProjects/360ISOD_evaluation/gt';

%select the test image dataset path
datasetPath = '/Users/zhangyi/Projects/MatlabProjects/360ISOD_evaluation/predicted';

%set the result Path
resPath = 'F:\FDP\New SSIM\failure case\Fbw\';
txtPath = 'F:\FDP\New SSIM\failure case\Fbw\result.txt'

fileID = fopen(txtPath,'wt');

imageSet = build_database(datasetPath,'.png');
for i = 1:imageSet.nclass
    name = imageSet.cname{i};
    
    %gt image
    if(strfind(name,'.jpg'))
        gt_s = regexp(name,'.jpg','split');
        [GT,map] = imread([gtPath char(gt_s(1)) '.png']);        
        if numel(size(GT))>2
            GT = rgb2gray(GT);
        end
        GT = logical(GT);
        if~isempty(map)
            figure();imshow(GT);
            GT = ~GT;
            figure();imshow(GT);title('reverse GT');
        end      
    else  %test image
        test_s = regexp(name,'_','split');
        lastName = char(test_s(2));
        FG = imread([datasetPath name]);
        
        if numel(size(FG))>2
            FG = rgb2gray(FG);
        end
        dFG = double(FG);
        dFG = reshape(mapminmax(dFG(:)',0,1),size(FG));
        %figure();imshow(dFG);
        
        %evaluate
        Q = original_WFb(dFG,GT);
        
        %[ivar,ovar] = Inner_var(dFG,GT);     
        %score = (1 - Q) + ivar ;
        score = Q;
        score = num2str(roundn(score,-4));
        score = FormateScore(score);
        
        resName = [char(resPath) char(gt_s(1)) '_' score '_Fwb_' char(test_s(2))];
        gtName = [char(resPath) char(gt_s(1)) '.png'];
        
        %figure();imshow(GT);
        imwrite(GT,gtName);
        imwrite(FG,resName);
        
        fprintf(fileID,'%s\n',score);
        
    end
end

fclose(fileID);



