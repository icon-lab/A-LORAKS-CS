%% example 3: Comparison of different LORAKS matrices and neighborhood sizes
clear;
close all;
clc
addpath('../.'); % Add the path containing the LORAKS functions
warning('off','MATLAB:pcg:tooSmallTolerance');

%% Multi-channel MPRAGE data
load MPRAGE_multi_channel % Load k-space data
kMask = kMaskRandACS7; % Random sampling with ACS data and acceleration factor = 7
undersampledData = kData.*repmat(kMask,[1 1 size(kData,3)]);
rGold = sqrt(sum(abs(fftshift(ifft2(ifftshift(kData)))).^2,3)); % Sum-of-squares gold standard image

max_iter = 2000; % Use a large number of iterations to compare convergence speeds
tol = 1e-2; % Use a modest convergence criterion to reduce noise amplification

%%
for trial = [1:24]
    if trial == 1 % C matrix, R = 2, no VCC
        LORAKS_type = 'C';
        R = 2;
        rank = 30;
        VCC = 0;
    elseif trial == 2 % C matrix, R = 3, no VCC
        LORAKS_type = 'C';
        R = 3;
        rank = 40;
        VCC = 0;
    elseif trial == 3 % C matrix, R = 4, no VCC
        LORAKS_type = 'C';
        R = 4;
        rank = 60;
        VCC = 0;
    elseif trial == 4 % C matrix, R = 5, no VCC
        LORAKS_type = 'C';
        R = 5;
        rank = 80;
        VCC = 0;
    elseif trial == 5 % S matrix, R = 2, no VCC
        LORAKS_type = 'S';
        R = 2;
        rank = 55;
        VCC = 0;
    elseif trial == 6 % S matrix, R = 3, no VCC
        LORAKS_type = 'S';
        R = 3;
        rank = 95;
        VCC = 0;
    elseif trial == 7 % S matrix, R = 4, no VCC
        LORAKS_type = 'S';
        R = 4;
        rank = 145;
        VCC = 0;
    elseif trial == 8 % S matrix, R = 5, no VCC
        LORAKS_type = 'S';
        R = 5;
        rank = 180;
        VCC = 0;
    elseif trial == 9 % W matrix, R = 2, no VCC
        LORAKS_type = 'W';
        R = 2;
        rank = 20;
        VCC = 0;
    elseif trial == 10 % W matrix, R = 3, no VCC
        LORAKS_type = 'W';
        R = 3;
        rank = 30;
        VCC = 0;
    elseif trial == 11 % W matrix, R = 4, no VCC
        LORAKS_type = 'W';
        R = 4;
        rank = 45;
        VCC = 0;
    elseif trial == 12 % W matrix, R = 5, no VCC
        LORAKS_type = 'W';
        R = 5;
        rank = 80;
        VCC = 0;
    elseif trial == 13 % C matrix, R = 2, with VCC
        LORAKS_type = 'C';
        R = 2;
        rank = 55;
        VCC = 1;
    elseif trial == 14 % C matrix, R = 3, with VCC
        LORAKS_type = 'C';
        R = 3;
        rank = 95;
        VCC = 1;
    elseif trial == 15 % C matrix, R = 4, with VCC
        LORAKS_type = 'C';
        R = 4;
        rank = 150;
        VCC = 1;
    elseif trial == 16 % C matrix, R = 5, with VCC
        LORAKS_type = 'C';
        R = 5;
        rank = 210;
        VCC = 1;
    elseif trial == 17 % S matrix, R = 2, with VCC
        LORAKS_type = 'S';
        R = 2;
        rank = 70;
        VCC = 1;
    elseif trial == 18 % S matrix, R = 3, with VCC
        LORAKS_type = 'S';
        R = 3;
        rank = 110;
        VCC = 1;
    elseif trial == 19 % S matrix, R = 4, with VCC
        LORAKS_type = 'S';
        R = 4;
        rank = 180;
        VCC = 1;
    elseif trial == 20 % S matrix, R = 5, with VCC
        LORAKS_type = 'S';
        R = 5;
        rank = 190;
        VCC = 1;
    elseif trial == 21 % W matrix, R = 2, with VCC
        LORAKS_type = 'W';
        R = 2;
        rank = 50;
        VCC = 1;
    elseif trial == 22 % W matrix, R = 3, with VCC
        LORAKS_type = 'W';
        R = 3;
        rank = 80;
        VCC = 1;
    elseif trial == 23 % W matrix, R = 4, with VCC
        LORAKS_type = 'W';
        R = 4;
        rank = 125;
        VCC = 1;
    elseif trial == 24 % W matrix, R = 5, with VCC
        LORAKS_type = 'W';
        R = 5;
        rank = 140;
        VCC = 1;
    end
    
    disp('********************************************************************');
    disp(['LORAKS_type = ' LORAKS_type ', R = ' num2str(R) ', VCC = ' num2str(VCC)]);
    disp('********************************************************************');
    
    % AC-LORAKS reconstruction with Eq. (13).  (exact data consistency)
    tic
    recon = AC_LORAKS(undersampledData, kMask, rank, R, LORAKS_type, [], [], tol, max_iter, VCC);
    time = toc;
    
    % Display results
    rSoS = sqrt(sum(abs(fftshift(ifft2(ifftshift(recon)))).^2,3));
    figure;
    imagesc(rSoS);
    axis equal;axis off;colormap(gray);
    caxis([0,1.3]);
    title(['LORAKS\_type = ' LORAKS_type ', R = ' num2str(R) ', VCC = ' num2str(VCC) ', NRMSE = ' num2str(norm(rGold(:)-rSoS(:))/norm(rGold(:))) ', time = ' num2str(time) ' seconds']);
    disp(' ');
end