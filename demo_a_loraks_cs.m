%% A-LORAKS-CS Demo
% Automated Parameter Selection for Accelerated MRI Reconstruction via
% Low-Rank Modeling of Local k-Space Neighborhoods (A-LORAKS-CS)
%
% This is a demo that shows an example implementation of A-LORAKS and
% A-LORAKS-CS over bSSFP data with multiple phase-cycled acquisitions and
% receiver coils. A-LORAKS-CS is a data-driven parameter tuning strategy 
% to automate hybrid PI-CS reconstructions. It employs singular value
% thresholding for the matrix rank selection in LORAKS, a SURE expression 
% for wavelet regularization weight selection and local standard deviation 
% method for total variation (TV) regularization weight selection. 
%
% This code requires SPIRiT (V0.3) and LORAKS (v2) libraries, and utilizes 
% parts of SparseMRI (V0.2) and SURE libraries.
% (https://people.eecs.berkeley.edu/~mlustig/Software.html)
% (https://mr.usc.edu/download/LORAKS2/)
% (https://candes.su.domains/software/SURE/data.html)
%
% The code is based on:
% Efe Ilicak, Emine Ulku Saritas, Tolga Çukur,
% Automated Parameter Selection for Accelerated MRI Reconstruction via 
% Low-Rank Modeling of Local k-Space Neighborhoods,
% Zeitschrift für Medizinische Physik, 2022.
% https://doi.org/10.1016/j.zemedi.2022.02.002

%% Retrospective Undersampling
% Add necessary folders to path
addpath(genpath("ESPIRiT"));
addpath(genpath("LORAKS V2.1"));
addpath(genpath("Utils"));

% Load fully sampled noiseless brain phantom data. For demo purposes, the
% data here contains a single slice with 2 phase cycles.
load('demo_data.mat'); 

% Add complex noise
snr=30;
rng(2202); % Set random number generator to a seed for predictable results
timg = ifft2c(kdata);
nlevel = max(abs(timg(:)))/snr./sqrt(2);
noise = nlevel*randn(size(kdata))+sqrt(-1)*nlevel*randn(size(kdata));
ndata = kdata+noise;

[PE1,PE2,RO,SC,SP] = size(ndata); % Phase Encodes, Readout (Fully Sampled), Coils, Phase Cycles (Multiple Acquisitions)

accel = 4; % Acceleration ratio

% Create sampling mask
FSR = 0.1; % Fully Sampled Region
[pdf,mask,actAcc] = genPDFandMask([PE1,PE2],accel,FSR); % Create an undersampling mask 
[calibSize, densComp] = getCalibSize(mask);

% Retrospectively undersample data
udata = ndata.*repmat(mask,[1 1 RO SC SP]);

%% Reconstruction
% Memory pre-allocation 
res_loraks = zeros(size(udata)); % A-LORAKS reconstruction
cs_loraks = zeros(size(udata)); % A-LORAKS-CS reconstruction
loraks_R = 3; % LORAKS k-space radius

% Go over multiple acquisitions and slices independently
for pcIdx = 1:SP
    for slcIdx = 1:RO
        
        data = squeeze(udata(:,:,slcIdx,:,pcIdx)); % Select acquisition
        
        % Estimate noise standard deviation
        XPre = ifft2c(data./pdf); % Previous solution (image domain)
        noiseSigma = zeros(1,size(XPre,3)); % Noise standard deviation
        % Find noise standard deviation from the image (do it only once)
        for pidx=1:size(XPre,3)
            noiseSigma(pidx) = find_sigma(XPre(:,:,pidx));
        end
        stdev = mean(noiseSigma);
        
        % A-LORAKS Reconstruction
        res_loraks(:,:,slcIdx,:,pcIdx) = aLoraks(data,mask,loraks_R,stdev); %res_loraks is in k-space
        
        % A-LORAKS-CS Reconstruction 
        cs_loraks(:,:,slcIdx,:,pcIdx) = autoCS(res_loraks(:,:,slcIdx,:,pcIdx),data,mask); % cs_loraks is in image domain
        
        % Display Progress
        fprintf('\nReconstruction of Acquisition: %d/%d, Slice: %d/%d completed.\n\n',pcIdx,SP,slcIdx,RO);
    end
end

% Combine receiver coils 
ref   = norm2d(sos(sos(ifft2c(kdata)))); 
zf    = norm2d(sos(sos(ifft2c(udata./pdf)))); 

alor   = norm2d(sos(sos(ifft2c(res_loraks))));
alorcs = norm2d(sos(sos(cs_loraks)));

recons = cat(3,ref,zf,alor,alorcs);
% Normalize and adjust data for better visualization
for rIdx=1:size(recons,3)
sLims = stretchlim(recons(:,:,rIdx),[0 0.99]);
recons(:,:,rIdx)  = imadjust(recons(:,:,rIdx),sLims,[0 1]);
end
%% Display Results
% Compute image quality metrics
for rIdx = 1:size(recons,3)
            RR = segMask.*recons(:,:,1); RRn = RR(~isnan(RR)); % Reference
            RE = segMask.*recons(:,:,rIdx); REn = RE(~isnan(RE)); % Recons
            psnr_vals(rIdx) = psnr(RRn,REn);
            [~,t_ssv] = ssim(RR,RE);
            ssim_vals(rIdx) =  nanmean(t_ssv(:))*100;
            nmse_vals(rIdx) = nmse(REn,RRn)*100;
end

fprintf('PSNR Results |  A-LORAKS   : %2.2f\n', psnr_vals(3));
fprintf('             |  A-LORAKS-CS: %2.2f\n\n', psnr_vals(4));
fprintf('SSIM Results |  A-LORAKS   : %2.2f\n', ssim_vals(3));
fprintf('             |  A-LORAKS-CS: %2.2f\n\n', ssim_vals(4));
fprintf('NMSE Results |  A-LORAKS   : %2.2f\n', nmse_vals(3));
fprintf('             |  A-LORAKS-CS: %2.2f\n\n', nmse_vals(4));

% Display Images
figure,
subplot(2,2,1),imshow3(recons(:,:,1)),title('Reference')
subplot(2,2,2),imshow3(recons(:,:,2)),title('ZF')
subplot(2,2,3),imshow3(recons(:,:,3)),title('A-LORAKS')
subplot(2,2,4),imshow3(recons(:,:,4)),title('A-LORAKS-CS')


