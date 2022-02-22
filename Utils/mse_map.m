function [msemap logmap] = mse_map(im,imref,mnorm);
% calculates mse error
% mnorm specifies type of image normalization prior to metric calculation
% mnorm=1 => normalize by mean (small deviations)
% mnorm=2 => normalize 98% range of intensity values (larger deviations)
if nargin < 3
    mnorm = 0; % default, no normalization
end

im = abs(im);
imref = abs(imref);
if mnorm == 1
    im = im./max(im(:));
    imref = imref./max(imref(:));
elseif mnorm == 2
    im = im./max(im(:));
    imref = imref./max(imref(:));
    im = imadjust(im);
    imref = imadjust(imref);
elseif mnorm == 3
    im = im./max(im(:));
    imref = imref./max(imref(:));
    imref = imadjust(imref);
    im = imhistmatch(im,imref);
end
% calc mse
msemap = abs(im-imref).^2;
logmap = 10*log10(msemap);
logmap(isinf(logmap)) = min(logmap(~isinf(logmap)));