function recon_loraks = aLoraks(kData,kMask,R,stdev)
% ALORAKS -- A-LORAKS: Parameter free LORAKS reconstruction.
%
% recon_loraks = aloraks(kData,kMask,R)
%
% Inputs:
%
% kData         = Undersampled k-space data (2D x coils)
% kMask         = Sampling mask used in the acquisition
% R             = Loraks neighborhood size
% stdev         = Standard deviation of noise 
%
% Outputs:
%
% recon_loraks  = Reconstructed k-space data
%
% Efe Ilicak
% 18/02/2021
%

%% Fixed Loraks parameters
LORAKS_type = 'S'; % Recommended
VCC = 0; % Recommended
alg = 4; % Recommended
tol = 1e-3; % Default
loraks_iter = 50;
loraks_lambda = 0;

%% Estimate optimal rank for given neighborhood size
MM = P_LORAKS_Operator(kData, kMask, R);
[~, S, ~] = svd(MM,'eco');
s = diag(S);
rank = sureLoraksRank(s, stdev, size(MM));

%% P-LORAKS reconstruction
recon_loraks = P_LORAKS(kData, kMask, rank, R, LORAKS_type, loraks_lambda, alg, tol, loraks_iter, VCC);

end