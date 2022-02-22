function [best_lambda,res,nextRange] = autoSparsity2(f1,f0,n,sigma,lmbd,cdata,mask)
%
% AUTO SPARSITY TERM Calculates the best lambda regularization term based  
% on SURE and optionally gives the reconstruction and the next search range
%
% Inputs:
% f1:       Prior solution (For initialization use ZF recon)
% f0:       2nd Prior Solution (For initialization use zeros)
% n:        Iteration Number (Used for Nesterov Opt. coefficients)
% sigma:    Noise standard deviation estimate
% lmbd:     Regulaziration term search range (external search range)
% cdata:    Data consistency term
% mask:     The sampling mask used for obtaining cdata
%
% Outputs:
% best_lambda: Optimal lambda value for reconstruction
% res:         Reconstruction for that optimal lambda value
% nextRange:   The next search range to be used in the next iteration

if nargin <4
    error('Noise standard deviation (sigma) should be provided');
end

if nargin <5 || isempty(lmbd)==1
    lmbd = linspace(2e-5,2e-1,100);
end

if nargin <6 || isempty(cdata)==1
    cdata = fft2c(f1);
end

if nargin <7
    mask = ones(size(f1));
end

% Make the data,mask and maps dyadic
pwr = nextpow2(size(f1,1)); % Find the next power of 2
sx1 = 2^pwr;
pwr = nextpow2(size(f1,2)); % Find the next power of 2
sx2 = 2^pwr;
% Make the data, consistency data, mask power of 2
f1 = ifft2c(zpad(fft2c(f1),sx1,sx2));
f0 = ifft2c(zpad(fft2c(f0),sx1,sx2));
cdata = zpad(double(cdata),sx1,sx2);
mask = zpad(mask,sx1,sx2);

a = 1+(n-1)/(n+2); 
b = -1*(n-1)/(n+2); 
gn = a*f1 + b*f0; % Linear combination of two prior solutions
hn1 = gn + (ifft2c((cdata-mask.*fft2c(gn)))); % Intermediate image

% Preallocation
Res = zeros(sx1,sx2,length(lmbd));
sure = zeros(length(lmbd),1);

% Search for the best l1 term
for k=1:length(lmbd)
    WOP = Wavelet('Daubechies',4,4);
    lambda = lmbd(k);
    tmp_res = WOP'*(softThresh(WOP*(hn1),lambda));
    Res(:,:,k)=tmp_res;
    dt = max(abs(hn1(:)))*1e-4; % Threshold for degree of freedom. Set as 10^-6 of max of intermediate images
    dof = sum(sum(abs((WOP*tmp_res)>dt))); % Degree of freedom = # non-zero coeffs. in wavelet coeffs of f(^n+1)
    sure(k) = (-2)*numel(cdata)*sigma^2+norm((tmp_res-hn1),'fro')^2+4*sigma^2*dof;
end
% figure,plot(lmbd,sure),drawnow;
[~,loc]=min(sure);
res = Res(:,:,loc);
best_lambda = lmbd(loc);

% Find the next search range for lambda values
lmp = (best_lambda-lmbd(1))/2; % Lower midpoint length
ump = (lmbd(end)-best_lambda)/2; % Upper midpoint length
mmp = 2*min(lmp,ump); % Minimum of midpoint lengths 
nextRange = logspace(log10(best_lambda-mmp),log10(best_lambda+mmp),10); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
