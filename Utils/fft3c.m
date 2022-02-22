function fdata = fft3c(data)
% Performs 3-dim Fourier Transform for 3D,4D and 5D matrices
sd = size(data);
fdata = zeros(sd);

if length(sd)==3
    ispace = data;
    fdata=fftshift(fftn(ifftshift(ispace)));
    
elseif length(sd)==4
    for cidx=1:sd(end)
        ispace = data(:,:,:,cidx);
        fdata(:,:,:,cidx)=fftshift(fftn(ifftshift(ispace)));
    end
    
elseif length(sd)==5
    for aidx=1:sd(4)
        for cidx=1:sd(5)
            ispace = data(:,:,:,aidx,cidx);
            fdata(:,:,:,aidx,cidx)=fftshift(fftn(ifftshift(ispace)));
        end
    end
end