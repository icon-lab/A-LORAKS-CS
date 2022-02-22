function [pdf,mask,actAcc] = genPDFandMask(pes,accel,acr)
% Generates probability density function and the sampling mask according
% for undersampling the data.
% Inputs:
%           pes:    Phase encoding dimension size + number of phase cycles for phase-cycled bSSFP imaging
%           accel:  Acceleration rate
%           acr:    Fully sampled auto calibration region ratio
% 
% Outputs:
% pdf:      Generated probability density function
% mask:     Generated sampling mask(s)
% actAcc:   Actual acceleration rate
%
% 08.10.2019 
% Efe Ilicak

% Generate PDF
po=2; % Start from the smallest possible polynomial order
gf = true;
while gf == true
    try
        pdf = genPDF([pes(1),pes(2)],po,(1/accel),1,acr,0);
        gf=false;
    catch
        po = po+1;
    end
end

if po~=2
    fprintf('\nWarning: Polynomial order has been changed to %d\n',po);
end
% Generate Sampling Mask
rng(21159); % Set random number generator to a seed for predictable results - I used to use "53212" as seed

if size(pes)<5; pes(5)=1;end
for k=1:pes(5) % Generate different sampling patterns for all phase cycles
[mask(:,:,k),~,actAcc(k)] = genSampling(pdf,1000,numel(pdf)*0.01);
end