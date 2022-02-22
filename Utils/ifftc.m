function res = ifftc(x,dim)
%res = ifftc(x,dim)
if nargin < 2 % I added this part (16/01/2019 - Ilicak)
    dim = 2;
end
res = sqrt(size(x,dim))*fftshift(ifft(ifftshift(x,dim),[],dim),dim);

