%% example 2: Multi-channel reconstruction examples
clear;
close all;
clc
addpath('../.'); % Add the path containing the LORAKS functions
warning('off','MATLAB:pcg:tooSmallTolerance');

%% Multi-channel MPRAGE data
load MPRAGE_multi_channel % Load k-space data

% Display gold standard coil images
figure;
imagesc([abs(reshape(fftshift(ifft2(ifftshift(kData))),256,[]));reshape(angle(fftshift(ifft2(ifftshift(kData))))+pi,256,[])/2/pi]);
axis equal;axis off;colormap(gray);
caxis([0,1]);
title('Gold Standard');

% Display sum-of-squares of gold standard coil images
rGold = sqrt(sum(abs(fftshift(ifft2(ifftshift(kData)))).^2,3));
figure;
imagesc(rGold);
axis equal;axis off;colormap(gray);
caxis([0,1.3]);
title('Gold Standard (sum of squares)');

lambdaSENSELORAKS = 1e-3;
for sampling = 1:4
    if sampling == 1
        %% Sampling strategy: Random with ACS
        str = 'random with ACS';
        kMask = kMaskRandACS7;
        rankLORAKS = 75;
        rankACLORAKS = 120;
        rankSENSELORAKS = 55;
    elseif sampling == 2
        %% Sampling strategy: Random (Calibrationless)
        str = 'random (calibrationless)';
        kMask = kMaskRand7;
        rankLORAKS = 75;
        rankSENSELORAKS = 55;
    elseif sampling == 3
        %% Sampling strategy: Uniform with ACS
        str = 'uniform with ACS';
        kMask = kMaskUnACS7;
        rankLORAKS = 75;
        rankACLORAKS = 110;
        rankSENSELORAKS = 60;
    elseif sampling == 4
        %% Sampling strategy: Partial Fourier with ACS
        str = 'partial Fourier with ACS';
        kMask = kMaskPFACS7;
        rankLORAKS = 90;
        rankACLORAKS = 200;
        rankSENSELORAKS = 75;
    end
    
    disp('********************************************************************');
    disp(['Sampling: ' str]);
    disp('********************************************************************');
    
    % Display sampling pattern
    figure;
    imagesc(kMask);
    axis equal;axis off;colormap(gray);
    undersampledData = kData.*repmat(kMask,[1 1 size(kData,3)]);
    title(['Sampling: ' str])
    
    % P-LORAKS reconstruction with Eq. (6).  (S-matrix, exact data consistency)
    tic
    recon = P_LORAKS(undersampledData, kMask, rankLORAKS);
    time = toc;
    
    % Display results
    rSoS = sqrt(sum(abs(fftshift(ifft2(ifftshift(recon)))).^2,3));
    figure;
    imagesc(rSoS);
    axis equal;axis off;colormap(gray);
    caxis([0,1.3]);
    title(['P-LORAKS, ' str ', NRMSE = ' num2str(norm(rGold(:)-rSoS(:))/norm(rGold(:))) ', time = ' num2str(time) ' seconds']);
    disp(' ');
    
    % AC-LORAKS reconstruction with Eq. (13).  (S-matrix, exact data consistency)
    if not(sampling == 2)
        tic
        recon = AC_LORAKS(undersampledData, kMask, rankACLORAKS);
        time = toc;
        
        % Display results
        rSoS = sqrt(sum(abs(fftshift(ifft2(ifftshift(recon)))).^2,3));
        figure;
        imagesc(rSoS);
        axis equal;axis off;colormap(gray);
        caxis([0,1.3]);
        title(['AC-LORAKS, ' str ', NRMSE = ' num2str(norm(rGold(:)-rSoS(:))/norm(rGold(:))) ', time = ' num2str(time) ' seconds']);
        disp(' ');
    end
    
    % SENSE-LORAKS reconstruction with Eq. (9) (S-matrix)
    tic
    recon = SENSE_LORAKS(undersampledData, kMask, coil_sens, rankSENSELORAKS,lambdaSENSELORAKS);
    time = toc;
    
    % Display results
    rSoS = sqrt(sum(abs(coil_sens.*repmat(recon,[1,1,size(coil_sens,3)])).^2,3));
    figure;
    imagesc(rSoS);
    axis equal;axis off;colormap(gray);
    caxis([0,1.3]);
    title(['SENSE-LORAKS, ' str ', NRMSE = ' num2str(norm(rGold(:)-rSoS(:))/norm(rGold(:))) ', time = ' num2str(time) ' seconds']);
    disp(' ');
end