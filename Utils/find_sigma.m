function sigma = find_sigma(img)
% Given an image, this function estimates the noise standard deviation from
% the finest scales of the image. This sigma estimate is obtained prior to
% other operations and it is only calculated once.

[~,CH,CV,CD] = dwt2(img,'db4'); % Obtain wavelet transform of image
FinestScale = [CH,CV,CD]; % Drop the low frequency bin
FinestScale(abs(FinestScale)==0)=NaN; % If there are 0 values, set them to NaN
% Estimating the noise standard deviation as equal to 1.4286 times the
% median absolute deviation of of finest scale wavelet coefficients.
% figure,imshow3(abs(FinestScale));
sigma = 1.4826*mad(FinestScale(:),1);
end