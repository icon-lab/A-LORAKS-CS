function [recon,alliters,all_WT,all_TV] = autoCS(kdata,acqdata,mask)
% AUTOCS -- Automatic compressed sensing reconstuction using wavelet
% soft-thresholding and total variation regularization.
%
% [recon,alliters,all_WT,all_TV] = autoCS(kdata,acqdata,mask)
%
% Inputs:
%
% kdata     = current kspace data (can be after Parallel Imaging)
% acqdata   = scanner acquired kspace data
% mask      = sampling mask used for acquiring the data
%
% 
% Outputs:
%
% recon     = reconstructed image
% alliters  = reconstructed images across iterations
% all_WT    = used soft-threshold values across iterations
% all_TV    = used total variaton values across iterations
%
% Efe Ilicak
% 18/02/2021
%
% Added diadic padding/cropping
% 02/03/2021

kdata = squeeze(kdata);
acqdata = squeeze(acqdata);

% find the closest diadic size for the images
[sx,sy,nc] = size(kdata);
ssx = 2^ceil(log2(sx));
ssy = 2^ceil(log2(sy));
ss = max(ssx, ssy);

kdata = zpad(kdata,ss,ss,nc);
acqdata = zpad(acqdata,ss,ss,nc);
mask = zpad(mask,ss,ss);



W = Wavelet('Daubechies',4,4);
LogRange = logspace(-5,-2,20); % Initial SURE range
iterNum = 50;


fn   = (ifft2c(kdata));
fne1 = zeros(size(fn));
alliters(:,:,1) = sos(fn);
CS_tol = 1e-3;

disp('CS Reconstruction');
for n = 1:iterNum
    
    X = W*(fn); % apply wavelet
    for cc=1:size(fn,3)
        noiseSigma2 = find_sigma(fn(:,:,cc)); %noiseSigma(cc)
        [best_lambda(cc,n),~,Range] = autoSparsity2(fn(:,:,cc),fne1(:,:,cc),n,noiseSigma2,LogRange,acqdata(:,:,cc),mask);
        if cc==size(fn,3)
            LogRange = Range; % Update range after finishing every element
        end
    end
    st = mean(best_lambda(:,n));
    all_WT(n)=st;
    [X1] = softThresh(X,st);
    X1 = W'*(X1); % get back the image
    X1 = ifft2c(fft2c(X1).*~mask + acqdata.*mask); % fix the data
%     X1 = fn;
    X2 = X1;
    for pidx=1:size(X2,3)
        img = X2(:,:,pidx);
        [TVWeight,TVMap] = autoTV(img,1,10);
        all_TV(pidx,n) = TVWeight;
        [resTV]=denoiseTV(img,[TVMap(:);TVMap(:)],5,0);
        X2(:,:,pidx) = resTV;
    end
    X2 = ifft2c(fft2c(X2).*~mask + acqdata.*mask); % fix the data


    % Update prior solutions
    fne1 = fn;
    fn = X2;
    
    
    recon = X2;
    alliters(:,:,n+1) = sos(X2);
    
    % Check convergence
    t = (norm(fne1(:,:)-fn(:,:))/norm(fn(:,:)));
    % display the status
    disp(['iter ' num2str(n) ', relative change in solution: ' num2str(t)]);
    
    if t < CS_tol
        disp('CS Convergence tolerance met: change in solution is small');
        recon = ifft2c(crop(fft2c(recon),sx,sy,nc)); % return to the original size
%         figure,imshow3([ref zf lor lw lwt]),title(sprintf('Iter %d',n)), drawnow
        break
    end
    
end

recon = ifft2c(crop(fft2c(recon),sx,sy,nc)); % return to the original size
