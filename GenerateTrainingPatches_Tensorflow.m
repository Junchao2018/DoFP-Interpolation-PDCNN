
%%% Generate the training data.

clear;close all;

addpath('utilities');

batchSize      = 128;        %%% batch size
dataName      = 'TrainingPatches_Tensorflow';
folder        = 'image';
% subfolder = {'T1','T2','T3','image04','image02','image06','image07','S2','S3','S4','T7',...
%     'T8','T9','T10','T11','T13'};
subfolder = {'T1','T2','T3','image04','image02','image06','image07'};
imagename = {'0.bmp','45.bmp','90.bmp','135.bmp'};
patchsize     = 40;
stride        = 10;
step          = 0;

count   = 0;



%% count the number of extracted patches
scales  = [1 0.9 0.8 0.7];
for i = 1 : length(subfolder)
    image = imread(fullfile(folder,subfolder{i},imagename{1})); % uint8
    if size(image,3)==3
        image = rgb2gray(image);
    end
    %[~, name, exte] = fileparts(filepaths(i).name); 
    for s = 1:4
        image_re = imresize(image,scales(s),'bicubic');
        [hei,wid,~] = size(image_re);
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches = ceil(count/batchSize)*batchSize;

disp([numPatches,batchSize,numPatches/batchSize]);

%pause;

inputs  = zeros(numPatches,patchsize, patchsize, 4, 'single'); % this is fast
outputs  = zeros(numPatches,patchsize, patchsize, 4, 'single'); % this is fast


tic;
for k = 1:4
    count   = 0;
for i = 1 : length(subfolder)
    
        image_ori = imread(fullfile(folder,subfolder{i},imagename{k})); % uint8
        %[~, name, exte] = fileparts(filepaths(i).name);
        
        if size(image_ori,3)==3
            image_ori = rgb2gray(image_ori);
        end
        %     end
        for s = 1:4
            image = imresize(image_ori,scales(s),'bicubic');
            %imagein = zeros(size(image),'uint8');
            imagein = image;
            h = size(image,1);
            w = size(image,2);
            if k == 1
                lh = length([1:2:h]);
                lw = length([1:2:w]);
                [X,Y] = meshgrid(1:0.5:lw,1:0.5:lh);
                [CX,CY] = meshgrid(1:lw,1:lh);
                hh = 2*lh-1;
                hw = 2*lw-1;
                imagein(1:hh,1:hw) = uint8(interp2(CX,CY,double(image(1:2:end,1:2:end)),X,Y,'cubic'));
            else if k ==2
                    lh = length([1:2:h]);
                    lw = length([2:2:w]);
                    [X,Y] = meshgrid(1:0.5:lw,1:0.5:lh);
                    [CX,CY] = meshgrid(1:lw,1:lh);
                    hh = 2*lh-1;
                    hw = 2*lw-1;
                    imagein(1:hh,2:hw+1) = uint8(interp2(CX,CY,double(image(1:2:end,2:2:end)),X,Y,'cubic'));
                else if k ==3
                        lh = length([2:2:h]);
                        lw = length([2:2:w]);
                        [X,Y] = meshgrid(1:0.5:lw,1:0.5:lh);
                        [CX,CY] = meshgrid(1:lw,1:lh);
                        hh = 2*lh-1;
                        hw = 2*lw-1;
                        imagein(2:hh+1,2:hw+1) = uint8(interp2(CX,CY,double(image(2:2:end,2:2:end)),X,Y,'cubic'));
                    else
                        lh = length([2:2:h]);
                        lw = length([1:2:w]);
                        [X,Y] = meshgrid(1:0.5:lw,1:0.5:lh);
                        [CX,CY] = meshgrid(1:lw,1:lh);
                        hh = 2*lh-1;
                        hw = 2*lw-1;
                        imagein(2:hh+1,1:hw) = uint8(interp2(CX,CY,double(image(2:2:end,1:2:end)),X,Y,'cubic'));
                    end
                end
            end
            for j = 1:1
                image_aug = data_augmentation(image, j);  % augment data
                image_augin = data_augmentation(imagein, j);  % augment data
                im_label  = im2single(image_aug); % single
                im_labelin  = im2single(image_augin); % single
                [hei,wid,~] = size(im_label);
                
                for x = 1+step : stride : (hei-patchsize+1)
                    for y = 1+step :stride : (wid-patchsize+1)
                        count       = count+1;
                        inputs(count,:, :, k)   = im_labelin(x : x+patchsize-1, y : y+patchsize-1,:);
                        outputs(count,:, :, k)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                    end
                end
            end
        end
    end
end
toc;

if count<numPatches
    pad = numPatches - count;
    inputs(count+1:end,:,:,:) = inputs(1:pad,:,:,:);
    outputs(count+1:end,:,:,:) = outputs(1:pad,:,:,:);
end

disp('-------Datasize-------')
disp([size(inputs,1),batchSize,size(inputs,1)/batchSize]);

if ~exist(dataName,'file')
    mkdir(dataName);
end

%%% save data
save(fullfile(dataName,['imdb_',num2str(patchsize),'_',num2str(batchSize)]), 'inputs','outputs','-v7.3')

