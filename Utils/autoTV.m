function [TVWeight,TVMap] = autoTV(img,ratio,dcs)
%
% AUTO TV Calculates the regularization term for total variation penalty.
%
% Input:
% img:      Image to calculate the TV penalty from.
% ratio:    TV weight ratio of non-edge/edge (=1/ratio). (Default: 10)
% dcs:      Constant divider term (Default: 13)
%
% Output:
% TVWeight: Weight for total variaton regularization term.
% TVMap:    Adaptive weight map.

% Filter Image
HSIZE = round((size(img))/25);
SIGMA = 2;
h = fspecial('gaussian', HSIZE, SIGMA);
tmp = imfilter(img,h); % Low pass filtered image

% Find and normalize local variance of filtered image
nsize(1) = round(size(img,1)/100)+1-mod(round(size(img,1)/100),2);
nsize(2) = round(size(img,2)/100)+1-mod(round(size(img,2)/100),2);

% nsize shouldn't be [1 1]. In that case stdfilt would give NaN
if nsize(1)==1 && nsize(2)==1
    if size(img,1)>=size(img,2)
        nsize(1)=3; % nsize should be an odd number
    else
        nsize(2)=3; 
    end
end

NHOOD = ones(nsize);
tmp = norm2d(abs(tmp));
LS = stdfilt(tmp,NHOOD); % Local standard deviation of filtered image
LSm = sqrt(LS); % Square root of local standard deviation map

% Designate TV regularization term as the 1/dcs of median of the map
TVWeight = (median(LSm(:))/(dcs));

se = strel('disk',1);
map = imdilate(LSm,se); % Dilate for smoother transition
map = (map-min(map(:)))./(max(map(:))-min(map(:))); % Normalize between 0&1 (edge=1)
%map = 1./((map.*(ratio-1))+1); % Distribute between 1 and ration (1=non-edge, ratio=edge)
map = (((ratio-1).*(abs(map-1)))+1); % Distribure between ratio and 1 (ratio=non-edge, 1=edge)
TVMap = TVWeight.*map;
