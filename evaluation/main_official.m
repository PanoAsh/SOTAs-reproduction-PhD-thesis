clear all; close all; %clc;

%% set your dataset path and saliency map result path.

CACHE = 'cache/';

DS = {'ECSSD', 'PASCAL-S', 'HKU-IS', 'DUTSTE' ,'SOD', 'DUT'}; %, 'SOCTE'

%%
MD_ALL = {'AMU17','BDMP','C2S','DCL16','DGRL',...
          'DHS','DLS2017','DS16','DSS17','ELD16',...
          'FSN17','GT','HS','KSR16','LEGS',...
          'MCDL','MDF15','MSRNet','NLDF17','PAGRN18',...
          'PiCANet','PiCANet-C','PiCANet-R','PiCANet-RC','RADF',...
          'RAS','RFCN','SRM17','UCF17','wCtr',...
          'CRPSD', 'DRFI','MAP','SBF','ASNet',...
          'RSD-r', 'WSS'
         };
     
postfix={'.png', '.jpg', '.png', '_dcl_crf.png', '.png',...{'AMU17','BDMP','C2S','DCL16','DGRL'}
         '.png', '.png', '.png', '_dss_crf.png', '_ELD.png',...{'DHS','DLS2017','DS16','DSS17','ELD16'}
         '.png', '.png', '_HS.png', '.png', '.png',...{'FSN17','GT','HS','KSR16','LEGS'}
         '.png', '_MDF.png', '.png', '_NLDF.png', '.png',...{'MCDL','MDF15','MSRNet','NLDF17','PAGRN18'}
         '.png', '.png', '.png', '.png', '.jpg',...{'PiCANet','PiCANet-C','PiCANet-R','PiCANet-RC','RADF'}
         '_ras.png', '.jpg', '.png', '.png', '_wCtr_Optimized.png',...{'RAS','RFCN','SRM17','UCF17','wCtr'}
         '_CRPSD.png','.png','.png','.png','.png', ...{'CRPSD', 'DRFI','MAP','SBF','ASNet'}
         '.png', '.png' ...{'RSD-r', 'WSS'}
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
    if isequal(method, 'GT')||isequal(method, 'PiCANet')...
            ||isequal(method, 'PiCANet-C')||isequal(method, 'PiCANet-R')
        continue;
    end
    for didx=1:length(DS),
        dataset = DS{didx};
        
        if exist([CACHE, sprintf('%s_%s.mat',method, dataset)], 'file')
            load([CACHE, sprintf('%s_%s.mat',method, dataset)]);
        else
            % path of ground truth maps
            gtPath = ['G:/DataSets/Salienct_Object/' dataset '/masks/'];
            % path where prediction results are stored
            salPath = ['G:/DataSets/Salienct_Object/Results/' dataset '/' method '/'];
            
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
%             Emeasure=zeros(1,imgNUM)-1;
            MAE=zeros(1,imgNUM)-1;

            %% calculate MAE and Smeasure
            tic;
            for i = 1:imgNUM

    %             fprintf('Evaluating: %d/%d\n',i,imgNUM);

                gt_name =  imgFiles(i+2).name;
                sal_name =  replace(imgFiles(i+2).name,'.png', postfix{midx});
                if ~exist([salPath sal_name], 'file')
                    continue;
                end

                %load gt
                gt = imread([gtPath gt_name]);

                if numel(size(gt))>2
                    gt = rgb2gray(gt);
                end
                
                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end
                
                %load salency
                sal  = imread([salPath sal_name]);
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

                %You can change the method of binarization method. As an example, here just use adaptive threshold.
            %     threshold =  2* mean(sal(:)) ;
            %     if ( threshold > 1 )
            %         threshold = 1;
            %     end
            %     Bi_sal = zeros(size(sal));
            %     Bi_sal(sal>threshold)=1;
            %     Emeasure(i) = Enhancedmeasure(Bi_sal,gt);

            end

            toc;

            %%
            Smeasure(Smeasure==-1) = [];
%             Emeasure(Emeasure==-1) = [];
            MAE(MAE==-1) = [];
            
            %%
            Sm = mean2(Smeasure);
            Fm = max(F_curve);
            % Em = mean2(Emeasure);
            mae = mean2(MAE);

            Sm_std = std2(Smeasure);
            % Em_std = std2(Emeasure);
            mae_std = std2(MAE);
            
            %%
            if (~isnan(Fm)||~isnan(mae))||~isnan(Sm)               
                save([CACHE, sprintf('%s_%s.mat',method, dataset)], ...
                    'Fm', 'Sm', 'Sm_std', 'mae', 'mae_std');
                save([CACHE, sprintf('%s_%s_all.mat',method, dataset)], ...
                    'F_curve', 'Smeasure', 'MAE');
            end
            
        end
%         fprintf('(%s %s Dataset) Fmeasure: %.3f; Smeasure: %.3f+%.3f; MAE: %.3f+%.3f\n', ...
%             method, dataset, Fm, Sm, Sm_std, mae, mae_std);
        fprintf('(%s %s Dataset) Fmeasure: %.3f; Smeasure: %.3f; MAE: %.3f\n', ...
            method, dataset, Fm, Sm, mae);

    end
end
