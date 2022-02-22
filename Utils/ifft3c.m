function data = ifft3c(fdata)
% Performs 3-dim inverse Fourier Transform for 3D,4D and 5D matrices
sd = size(fdata);
data = zeros(sd);

if length(sd)==3
    kspace = fdata;
    data=fftshift(ifftn(ifftshift(kspace)));
    
elseif length(sd)==4
    for cidx=1:sd(end)
        kspace = fdata(:,:,:,cidx);
        data(:,:,:,cidx)=fftshift(ifftn(ifftshift(kspace)));
    end
    
elseif length(sd)==5
    for aidx=1:sd(4)
        for cidx=1:sd(end)
            kspace = fdata(:,:,:,aidx,cidx);
            data(:,:,:,aidx,cidx)=fftshift(ifftn(ifftshift(kspace)));
        end
    end
end