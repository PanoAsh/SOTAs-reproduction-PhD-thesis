clear all; close all; %clc;

%% set your dataset path and saliency map result path.

CACHE = 'cache/';

DS = { '360ISOD'}; %, 'SOCTE'

%%
MD_ALL = {'360ISOD'
         };
     
postfix={'.png'...{'360ISOD'}
        
        }; 
% 37 - 3 - 1 - 1 = 32

%% 
% MD_ALL = {'ASNet'};
% postfix={'.png'};

%%
targetIsFg = true; 
targetIsHigh = true;

for midx=1:length(MD_ALL),
    method = MD_ALL{midx};
    
    for didx=1:length(DS),
        dataset = DS{didx};
        
        if exist([CACHE, sprintf('%s_%s.mat',method, dataset)], 'file')
            load([CACHE, sprintf('%s_%s.mat',method, dataset)]);
        else
            % path of ground truth maps
            gtPath = '/Users/zhangyi/Projects/MatlabProjects/360ISOD_evaluation/gt';
            % path where prediction results are stored
            salPath = '/Users/zhangyi/Projects/MatlabProjects/360ISOD_evaluation/predicted';
            
            if ~exist(salPath, 'dir')
                fprintf('%s %s not exist.\n', dataset, method);
                continue;
            end
            
            %% calculate F-max
            [~, ~, ~, F_curve, ~] = DrawPRCurve(salPath, postfix{midx}, gtPath, '.png', targetIsFg, targetIsHigh);

            %% obtain the total number of image (ground-truth)
            imgFiles = dir(gtPath);
            imgNUM = length(imgFiles)-2;
            if imgNUM<0
                continue;
            end

            %% evaluation score initilization.
            Smeasure=zeros(1,imgNUM)-1;
            Emeasure=zeros(1,imgNUM)-1;
            MAE=zeros(1,imgNUM)-1;
            wFmeasure=zeros(1,imgNUM)-1;

            %% calculate MAE and Smeasure
            tic;
            for i = 1:imgNUM   
                

    %             fprintf('Evaluating: %d/%d\n',i,imgNUM);

                gt_name =  imgFiles(i+2).name;
                sal_name =  replace(imgFiles(i+2).name,'.png', postfix{midx});
               

                %load gt
                gt = imread(fullfile(gtPath, gt_name));
                
                if numel(size(gt))>2
                    gt = rgb2gray(gt);
                end
                
                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end
                
                %load salency
                sal  = imread(fullfile(salPath, sal_name));
                MAE(i) = CalMAE(sal, gt);
                
                %check size
                if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                    sal = imresize(sal,size(gt));
                    imwrite(sal,[salPath sal_name]);
                    % fprintf('Error occurs in the path: %s!!!\n', [salPath sal_name]);
                end
                
                %--------------------
                sal = im2double(sal(:,:,1));

                %normalize sal to [0, 1]
                sal = reshape(mapminmax(sal(:)',0,1),size(sal));

                Smeasure(i) = StructureMeasure(sal,logical(gt)); 
                
                wFmeasure(i) = weightFmeasure(sal,logical(gt));
          
                %You can change the method of binarization method. As an example, here just use adaptive threshold.
                threshold =  2* mean(sal(:)) ;
                if ( threshold > 1 )
                    threshold = 1;
                end
                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold)=1;
                Emeasure(i) = Enhancedmeasure(Bi_sal,gt);
                 
                % to debug
                gt_show = [gt_name(1:3) '.jpg'];
                imwrite(gt, gt_show);
                sal_show = [sal_name(1:3) '_MAE_' num2str(MAE(i)) '_S_' num2str(Smeasure(i)) '_E_' num2str(Emeasure(i)) '_wF_' num2str(wFmeasure(i)) '.png'];
                imwrite(sal, sal_show);
                
            end

            toc;

            %%
            Smeasure(Smeasure==-1) = [];
         
            wFmeasure(wFmeasure==-1) = [];
            
            Emeasure(Emeasure==-1) = [];
            
            MAE(MAE==-1) = [];
            
            %%
            Sm = mean2(Smeasure);
            Fm = max(F_curve);
            Em = mean2(Emeasure);
            mae = mean2(MAE);
            wFm = mean2(wFmeasure);

            Sm_std = std2(Smeasure);
            Em_std = std2(Emeasure);
            mae_std = std2(MAE);
            wFm_std = std2(wFmeasure);
            
            %%
            if (~isnan(Fm)||~isnan(mae))||~isnan(Sm)||~isnan(Em)||~isnan(wFm)               
                save([CACHE, sprintf('%s_%s.mat',method, dataset)], ...
                    'Fm', 'Sm', 'Sm_std', 'mae', 'mae_std', 'Em', 'Em_std', 'wFm', 'wFm_std');
                save([CACHE, sprintf('%s_%s_all.mat',method, dataset)], ...
                    'F_curve', 'Smeasure', 'MAE', 'Emeasure', 'wFmeasure');
            end
            
        end
%         fprintf('(%s %s Dataset) Fmeasure: %.3f; Smeasure: %.3f+%.3f; MAE: %.3f+%.3f\n', ...
%             method, dataset, Fm, Sm, Sm_std, mae, mae_std);
        fprintf('(%s %s Dataset) Fmeasure: %.3f; Smeasure: %.3f; MAE: %.3f; Emeasure: %.3f; wFmeasure: %.3f\n', ...
            method, dataset, Fm, Sm, mae, Em, wFm);

    end
end
