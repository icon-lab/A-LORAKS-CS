%% example 1: Single channel reconstruction examples
clear;
close all;
clc
addpath('../.');  % Add the path containing the LORAKS functions
warning('off','MATLAB:pcg:tooSmallTolerance');

%% single-channel T2-weighted data
load T2_single_channel % Load k-space data

% Display gold standard
figure;
imagesc([abs(fftshift(ifft2(ifftshift(kData)))),(angle(fftshift(ifft2(ifftshift(kData))))+pi)/2/pi]);
axis equal;axis off;colormap(gray);
caxis([0,1]);
title('Gold Standard');

for sampling = 1:4
    if sampling == 1
        %% Sampling strategy: Random with ACS
        str = 'random with ACS';
        kMask = kMaskRandACS2;
        rankLORAKS = 45;
        rankACLORAKS = 45;
    elseif sampling == 2
        %% Sampling strategy: Random (Calibrationless)
        str = 'random (calibrationless)';
        kMask = kMaskRand2;
        rankLORAKS = 45;
    elseif sampling == 3
        %% Sampling strategy: Uniform with ACS
        str = 'uniform with ACS';
        kMask = kMaskUnACS2;
        rankLORAKS = 40;
        rankACLORAKS = 45;
    elseif sampling == 4
        %% Sampling strategy: Partial Fourier with ACS
        str = 'partial Fourier with ACS';
        kMask = kMaskPFACS2;
        rankLORAKS = 40;
        rankACLORAKS = 35;
    end
    
    disp('********************************************************************');
    disp(['Sampling: ' str]);
    disp('********************************************************************');
    
    % Display sampling pattern
    figure;
    imagesc(kMask);
    axis equal;axis off;colormap(gray);
    undersampledData = kData.*kMask;
    title(['Sampling: ' str])
    
    % LORAKS reconstruction with Eq. (6).  (S-matrix, exact data consistency)
    tic
    recon = P_LORAKS(undersampledData, kMask, rankLORAKS);
    time = toc;
    
    % Display results
    figure;
    imagesc([abs(fftshift(ifft2(ifftshift(recon)))),(angle(fftshift(ifft2(ifftshift(recon))))+pi)/2/pi]);
    axis equal;axis off;colormap(gray);
    caxis([0,1]);
    title(['LORAKS, ' str ', NRMSE = ' num2str(norm(recon(:)-kData(:))/norm(kData(:))) ', time = ' num2str(time) ' seconds']);
    disp(' ');
    
    % AC-LORAKS reconstruction with Eq. (13).  (S-matrix, exact data consistency)
    if not(sampling == 2)
        tic
        recon = AC_LORAKS(undersampledData, kMask, rankACLORAKS);
        time = toc;
        
        % Display results
        figure;
        imagesc([abs(fftshift(ifft2(ifftshift(recon)))),(angle(fftshift(ifft2(ifftshift(recon))))+pi)/2/pi]);
        axis equal;axis off;colormap(gray);
        caxis([0,1]);
        title(['AC-LORAKS, ' str ', NRMSE = ' num2str(norm(recon(:)-kData(:))/norm(kData(:))) ', time = ' num2str(time) ' seconds']);
        disp(' ');
    end
end