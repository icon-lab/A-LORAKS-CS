%% example 5: Comparison of different regularization parameters
clear;
close all;
clc
addpath('../.'); % Add the path containing the LORAKS functions

%% Multi-channel MPRAGE data
load MPRAGE_multi_channel % Load k-space data
kMask = kMaskRandACS7; % Random sampling with ACS data and acceleration factor = 7
undersampledData = kData.*repmat(kMask,[1 1 size(kData,3)]);
rGold = sqrt(sum(abs(fftshift(ifft2(ifftshift(kData)))).^2,3)); % Sum-of-squares gold standard image

%%
rank = 120;
for lambda = [0,logspace(-5,2,8)]
    
    disp('********************************************************************');
    disp(['lambda = ' num2str(lambda)])
    disp('********************************************************************');
       
    % AC-LORAKS reconstruction with Eq. (13).  (exact data consistency)
    tic
    recon = AC_LORAKS(undersampledData, kMask, rank, [], [], lambda);
    time = toc;
    
    % Display results
    rSoS = sqrt(sum(abs(fftshift(ifft2(ifftshift(recon)))).^2,3));
    figure;
    imagesc(rSoS);
    axis equal;axis off;colormap(gray);
    caxis([0,1.3]);
    title(['lambda = ' num2str(lambda) ', NRMSE = ' num2str(norm(rGold(:)-rSoS(:))/norm(rGold(:))) ', time = ' num2str(time) ' seconds']);
    disp(' ');
end