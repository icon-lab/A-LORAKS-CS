function [recon] = SENSE_LORAKS(kData, kMask, coil_sens, rank, lambda, R, LORAKS_type, alg, tol, max_iter)
% This function provides capabilities to solve multi-channel SENSE-LORAKS
% reconstruction problems using the formulation from Eq. (9) from the
% technical report:
%
% [1] T. H. Kim, J. P. Haldar.  LORAKS Software Version 2.0:
%     Faster Implementation and Enhanced Capabilities.  University of Southern
%     California, Los Angeles, CA, Technical Report USC-SIPI-443, May 2018.
%
% The problem formulation implemented by this function was originally reported
% in:
%
% [2] T. H. Kim, J. P. Haldar. LORAKS makes better SENSE: Phase?constrained
%     partial fourier SENSE reconstruction without phase calibration. Magnetic
%     Resonance in Medicine 77:1021-1035, 2017.
%
% *********************
%   Input Parameters:
% *********************
%
%    kData: A 3D (size N1 x N2 x Nc) array of measured k-space data to be
%           reconstructed.  The first two dimensions correspond to k-space
%           positions, while the third dimension corresponds to the channel
%           dimension for parallel imaging.  Unsampled data samples should be
%           zero-filled.
%
%    kMask: A binary mask of size N1 x N2 that corresponds to the same k-space
%           sampling grid used in kData.   Each entry has value 1 if the
%           corresponding k-space location was sampled and has value 0 if that
%           k-space location was not measured.
%
%    coil_sens: A 3D (size N1 x N2 x Nc) array of estimated coil sensitivity
%               profiles.
%
%    rank:  The matrix rank value used to define the non-convex regularization
%           penalty from Eq. (2) of Ref. [1].
%
%    lambda: The regularization parameter from Eq. (9). 
%
%    R: The k-space radius used to construct LORAKS neighborhoods. If not
%       specified, the software will use R=3 by default.
%
%    LORAKS_type: A string that specifies the type of LORAKS matrix that will
%                 be used in reconstruction.  Possible options are: 'C', 'S',
%                 and 'W'.  If not specified, the software will use
%                 LORAKS_type='S' by default.
%
%    alg: A parameter that specifies which algorithm the software should use for
%         computation. There are four different options:
%            -alg=1: This choice will use the additive half-quadratic algorithm,
%                    as described in Eq. (22) of Ref. [1].
%            -alg=2: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (30) of Ref. [1].
%                    This version does NOT use FFTs.
%            -alg=3: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (30) of Ref. [1].
%                    This version uses FFTs without approximation, as in
%                    Eq. (35) of Ref. [1].
%            -alg=4: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (30) of Ref. [1].
%                    This version uses FFTs with approximation, as in Eq. (36)
%                    of Ref. [1].
%         If not specified, the software will use alg=4 by default.
%
%    tol: A convergence tolerance.  The computation will halt if the relative
%         change (measured in the Euclidean norm) between two successive
%         iterates is small than tol.  If not specified, the software will use
%         tol=1e-3 by default.
%
%    max_iter: The computation will halt if the number of iterations exceeds
%              max_iter.  If not specified, the software will default to using
%              max_iter=1000 for the additive half-quadratic algorithm (alg=1),
%              and will use max_iter=50 for the multiplicative half-quadratic
%              algorithms (alg=2,3, or 4).
%
% **********************
%   Output Parameters:
% **********************
%
%    recon: The N1 x N2 reconstructed image.
%
% This software is available from
% <a href="matlab:web('http://mr.usc.edu/download/LORAKS2/')">http://mr.usc.edu/download/LORAKS2/</a>.
% As described on that page, use of this software (or its derivatives) in
% your own work requires that you at least cite [1] and [2].
%
% V2.0 Tae Hyung Kim and Justin P. Haldar 5/7/2018
% V2.1 Tae Hyung Kim and Justin P. Haldar 6/10/2019
%
% This software is Copyright ©2018 The University of Southern California.
% All Rights Reserved. See the accompanying license.txt for additional
% license information.

%% Parameter settings
if not(exist('R','var')) || isempty(R)
    R = 3;
end

if not(exist('LORAKS_type','var')) || isempty(LORAKS_type) || strcmpi(LORAKS_type,'S')
    LORAKS_type = 1;
elseif strcmpi(LORAKS_type,'C')
    LORAKS_type = 2;
elseif strcmpi(LORAKS_type,'W')
    LORAKS_type = 3;
else
    error('Error: Invalid LORAKS_type');
end

if not(exist('rank','var')) || isempty(rank)
    error('Error: Invalid rank');
end

if not(exist('lambda','var')) || isempty(lambda)
    error('Error: lambda is required');
end

if not(exist('alg','var')) || isempty(alg)
    alg = 4;
end

if not(exist('tol','var')) || isempty(tol)
    tol = 1e-3;
end

if not(exist('max_iter','var')) || isempty(max_iter)
    if alg == 1
        max_iter = 1000;
    else
        max_iter = 50;
    end
end


%% Data settings
[N1, N2, Nc] = size(kData);

kMask = repmat(kMask, [1 1 Nc]);

data = vect(kData.*kMask);

%% SENSE operators
Ah = @(x) vect(sum(conj(coil_sens).*ift2(kMask.*reshape(x,[N1 N2 Nc])),3));
AhA = @(x) vect(sum(conj(coil_sens).*ift2(kMask.*ft2(coil_sens.*repmat(reshape(x,[N1 N2]),[1 1 Nc]))),3));

%% SENSE reconstruction (for initialization)
[recon,~,~,~] = pcg(@(x) AhA(x), Ah(data), 1e-4, 20);
sense_recon = reshape(recon, [N1 N2]);

%% SENSE-LORAKS Initialization
disp('SENSE-LORAKS Reconstruction');

Ahd = Ah(data);
z = sense_recon(:); %initialization as SENSE recon


[in1,in2] = meshgrid(-R:R,-R:R);
idx = find(in1.^2+in2.^2<=R^2);
patchSize = numel(idx);

% sizeM: LORAKS matrix size
if LORAKS_type == 1 % S matrix
    sizeM = [patchSize*2, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2];
elseif LORAKS_type == 2 % C matrix
    sizeM = [patchSize, (N1-2*R-even(N1))*(N2-2*R-even(N2))];
elseif LORAKS_type == 3 % W matrix
    sizeM = [patchSize, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2];
end
    
if (min(sizeM(1)*Nc, sizeM(2)) < rank)
    error('Rank parameter is too large (can''t be larger than the matrix dimensions)');
end


% k-space weights for W matrix
weights = [];
if LORAKS_type == 3
    % Fourier weight for horizontal spatial difference
    W1 = repmat(1:N2,[N1 1 Nc])-1-N2/2+0.5*rem(N2,2);
    % Fourier weight for vertical spatial difference
    W2 = repmat(reshape(1:N1,N1,1),[1 N2 Nc])-1-N1/2+0.5*rem(N1,2);
    weights = cat(4,W1,W2);
end


% P_M: LORAKS matrix constructor
% Ph_M: its adjoint
% mm: diagonal element of Ph_M(P_M) matrix
P_M = @(x) LORAKS_operators(x,N1,N2,Nc,R,LORAKS_type,weights);
Ph_M = @(x) LORAKS_operators(x,N1,N2,Nc,R,-LORAKS_type,weights);
mm = repmat(LORAKS_operators(LORAKS_operators(ones(N1*N2,1,'like',kData)...
            ,N1,N2,1,R,LORAKS_type,weights),N1,N2,1,R,-LORAKS_type,weights), [Nc 1]);


if LORAKS_type == 1 && alg == 3
    delta = zeros([N1+2*R N2+2*R],'like',kData);
    delta(1,1) = 1;
    ph = fft2(circshift(delta, [-rem(N1,2) -rem(N2,2)]));
    Ic = @(x) x - conj(x).*repmat(ph,[1 1 1 size(x,4)]);  % multiply shift by -1 -> linear phase
end


if alg ~=4
    tmp = lambda*reshape(mm,[N1 N2 Nc]);
else 
    tmp = 0;
end

% SENSE-LORAKS forward model and its adjoint
EhE = @(x) vect(sum(conj(coil_sens).*ift2((kMask + tmp).*ft2(coil_sens.*repmat(reshape(x, [N1 N2]),[1 1 Nc]))),3));

B = @(x) vect(ft2(coil_sens.*repmat(reshape(x, [N1 N2]),[1 1 Nc])));           % SENSE encoding
Bh = @(x)  vect(sum(conj(coil_sens).*ift2(reshape(x,[N1, N2, Nc])),3));         % Adjoint of SENSE encoding

ZD = @(x) padarray(reshape(x,[N1 N2 Nc]),[2*R, 2*R], 'post');
ZD_H = @(x) x(1:N1,1:N2,:,:);
MC = @(x) sum(x,3);
MC_H = @(x) repmat(x,[1 1 Nc]);
T = false(N1,N2);
T(R+1+(~rem(N1,2)):N1-R, R+1+(~rem(N2,2)):N2-R) = 1;
T = padarray(T,[R R]);

%% Recon

for iter = 1:max_iter
    z_cur = z;
    pz = B(z);
    MM = P_M(pz);            

    if alg == 1 % Additive
        pmm = svd_left(MM,rank)'; % subspace
        MMr = Ph_M(pmm'*pmm*MM);
        Bhr = Bh(MMr);
        LhL = @(x) 0;

    elseif alg == 2 % Multiplicative
        pmm = svd_left(MM,rank)'; % subspace
        LhL = @(x) -Bh(Ph_M(pmm'*(pmm*P_M(B(x)))));
        Bhr = 0;

    elseif alg == 3  % Multiplicative with FFT
        pmm = svd_left(MM,rank)'; % subspace
        Bhr = 0;

        if LORAKS_type == 1 % S
            nf = size(pmm,1);
            pmm = reshape(pmm,[nf, patchSize, 2*Nc]);
            pss_h = reshape(pmm(:,:,1:2:end)+1j*pmm(:,:,2:2:end),[size(pmm,1), size(pmm,2)*Nc]);
            ffilt = fft2(padfilt(pss_h,N1,N2,Nc,R));
            LhL = @(x)  -Bh(ZD_H(sum(ifft2(conj(ffilt).*MC_H(Ic(fft2(repmat(T,[1 1 1 nf]).*ifft2(Ic(MC(ffilt.*fft2(repmat(ZD(B(x)),[1 1 1 nf]))))))))),4)));
        elseif LORAKS_type == 2 % C
            nf = size(pmm,1);
            ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
            LhL = @(x)  -Bh(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(B(x)),[1 1 1 nf])))))),4))));
        elseif LORAKS_type == 3 % W
            nf = size(pmm,1);
            ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
            LhL = @(x)  Bh(-conj(W1(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W1(:).*B(x)),[1 1 1 nf])))))),4))))...
                -conj(W2(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W2(:).*B(x)),[1 1 1 nf])))))),4)))));
        end
    elseif alg == 4 % Multiplicative with FFT (fast approximate)
        Um = svd_left(MM);
        nmm = Um(:,rank+1:end)'; % null space
        Bhr = 0;

        if LORAKS_type == 1 % S
            nf = size(nmm,1);
            nmm = reshape(nmm,[nf, patchSize, 2*Nc]);
            nss_h = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, patchSize*Nc]);
            Nis = filtfilt(nss_h,'C',N1,N2,Nc,R);
            Nis2 = filtfilt(nss_h,'S',N1,N2,Nc,R);
            LhL = @(x) 2*Bh((ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(B(x))),[1 1 1 Nc]),3))))) ...
                -(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(B(x)))),[1 1 1 Nc]),3))))));
        elseif LORAKS_type == 2 % C
            Nic = filtfilt(nmm,'C',N1,N2,Nc,R);
            LhL = @(x) Bh(ZD_H(ifft2(squeeze(sum(Nic.*repmat(fft2(ZD(B(x))),[1 1 1 Nc]),3)))));
        elseif LORAKS_type == 3 % W
            Niw = filtfilt(nmm,'C',N1,N2,Nc,R);
            LhL = @(x) Bh(conj(W1(:)).*vect(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W1(:).*B(x))),[1 1 1 Nc]),3)))))...
                + conj(W2(:)).*vect(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W2(:).*B(x))),[1 1 1 Nc]),3))))));
        end
    else
        error('Error: Invalid alg');
    end
    
    % data fitting
    M = @(x) EhE(x) + lambda*LhL(x);
    [z,~] = pcg(M, Ahd + lambda*Bhr);
    
    t = (norm(z_cur-z)/norm(z));
    
    % display the status
    if ~rem(iter,1)
        disp(['iter ' num2str(iter) ', relative change in solution: ' num2str(t)]);
    end
    
    % check for convergence
    if t < tol
        disp('Convergence tolerance met: change in solution is small');
        break;
    end
    
end
if iter==max_iter
    disp('Maximum number of iterations reached');
end

recon = reshape(z, [N1 N2]);
end

%%
function [ out ] = ft2( in )
out = fftshift(fft2(ifftshift(in)));
end

%%
function [ out ] = ift2( in )
out = fftshift(ifft2(ifftshift(in)));
end
%%
function result = even(int)
result = not(rem(int,2));
end

%%
function out = vect( in )
out = in(:);
end

%%
function U = svd_left(A, r)
% Left singular matrix of SVD (U matrix)
% parameters: matrix, rank (optional)
if nargin < 2
    [U,E] = eig(A*A'); % surprisingly, it turns out that this is generally faster than MATLAB's svd, svds, or eigs commands
    [~,idx] = sort(abs(diag(E)),'descend');
    U = U(:,idx);
else
    [U,~] = eigs(double(A*A'), r);
    U = cast(U, 'like', A);
end
end

%%
function result = LORAKS_operators(x, N1, N2, Nc, R, LORAKS_type, weights)
if LORAKS_type == 1     % S matrix
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros(patchSize,2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2,'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);

            tmp = x(Ind,:)-x(Indp,:);
            result(:,1,:,k) = real(tmp);
            result(:,2,:,k) = -imag(tmp);

            tmp = x(Ind,:)+x(Indp,:);
            result(:,1,:,k+end/2) = imag(tmp);
            result(:,2,:,k+end/2) = real(tmp);
        end
    end

    result = reshape(result, patchSize*Nc*2,(N1-2*R-even(N1))*(N2-2*R-even(N2))*2);

elseif LORAKS_type == -1 % S^H (Hermitian adjoint of of S matrix) 
    result = zeros(N1*N2,Nc,'like',x);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));

    x = reshape(x, [patchSize*2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2]);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);

            result(Ind,:) = result(Ind,:) + complex(x(1:patchSize,:,k) + x(patchSize+1:2*patchSize,:,nPatch+k), x(1:patchSize,:,nPatch+k) - x(patchSize+1:2*patchSize,:,k)); 
            result(Indp,:) = result(Indp,:) + complex( - x(1:patchSize,:,k) + x(patchSize+1:2*patchSize,:,nPatch+k), x(1:patchSize,:,nPatch+k) + x(patchSize+1:2*patchSize,:,k)); 

        end
    end

    result = vect(result);

elseif LORAKS_type == 2  % C matrix
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros(patchSize*Nc,(N1-2*R-even(N1))*(N2-2*R-even(N2)),'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(:,k) = vect(x(ind,:));
        end
    end

elseif LORAKS_type == -2 % C^H (Hermitian adjoint of of C matrix) 
    result = zeros(N1*N2,Nc,'like',x);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(ind,:) = result(ind,:)+ reshape(x(:,k),patchSize,Nc);
        end
    end
    result = vect(result);

elseif LORAKS_type == 3  % W matrix
    W1 = weights(:,:,1:Nc,1);
    W2 = weights(:,:,1:Nc,2);
    
    W1x = reshape(W1(:).*x(:),N1*N2,Nc);
    W2x = reshape(W2(:).*x(:),N1*N2,Nc);
    
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));
    result = zeros(patchSize*Nc,(N1-2*R-even(N1))*(N2-2*R-even(N2))*2,'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(:,k) = vect(W1x(ind,:));
            result(:,nPatch+k) = vect(W2x(ind,:));
        end
    end
    
elseif LORAKS_type == -3  % W^H (Hermitian adjoint of of W matrix)
    W1 = reshape(weights(:,:,1:Nc,1),[N1*N2,Nc]);
    W2 = reshape(weights(:,:,1:Nc,2),[N1*N2,Nc]);
    
    result1 = zeros(N1*N2,Nc,'like',x);
    result2 = zeros(N1*N2,Nc,'like',x);
    
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result1(ind,:) = result1(ind,:)+ reshape(x(:,k),patchSize,Nc);
            result2(ind,:) = result2(ind,:)+ reshape(x(:,nPatch+k),patchSize,Nc);
        end
    end
    result = vect(conj(W1).*result1 + conj(W2).*result2);
end
end

%%
function padded_filter = padfilt(ncc, N1, N2, Nc, R)
% zeropadding of the filter     (for alg=3)

fltlen = size(ncc,2)/Nc;    % filter length
numflt = size(ncc,1);       % number of filters

% LORAKS kernel is circular.
% Following indices account for circular elements in a square patch
[in1,in2] = meshgrid(-R:R,-R:R);
idx = find(in1.^2+in2.^2<=R^2);
in1 = in1(idx)';
in2 = in2(idx)';

ind = sub2ind([2*R+1, 2*R+1],R+1+in1,R+1+in2);

patch = zeros([(2*R+1)*(2*R+1), Nc, numflt],'like',ncc);
patch(ind,:,:) = reshape(ncc.', [fltlen Nc numflt]);
patch = reshape(patch,[2*R+1,2*R+1, Nc, numflt]);

patch = flip(flip(patch,1),2);
padded_filter = padarray(patch, [N1-1 N2-1],'post');
end

%%
function Nic = filtfilt(ncc, opt, N1, N2, Nc, R )
% Fast computation of zero phase filtering (for alg=4)
fltlen = size(ncc,2)/Nc;    % filter length
numflt = size(ncc,1);       % number of filters

% LORAKS kernel is circular.
% Following indices account for circular elements in a square patch
[in1,in2] = meshgrid(-R:R,-R:R);
idx = find(in1.^2+in2.^2<=R^2);
in1 = in1(idx)';
in2 = in2(idx)';

ind = sub2ind([2*R+1, 2*R+1],R+1+in1,R+1+in2);

filtfilt = zeros((2*R+1)*(2*R+1),Nc,numflt,'like',ncc);
filtfilt(ind,:,:) = reshape(permute(ncc,[2,1]),[fltlen,Nc,numflt]);
filtfilt = reshape(filtfilt,(2*R+1),(2*R+1),Nc,numflt);

cfilt = conj(filtfilt);

if opt == 'S'       % for S matrix
    ffilt = conj(filtfilt);
else                % for C matrix
    ffilt = flip(flip(filtfilt,1),2);
end

ccfilt = fft2(cfilt,4*R+1, 4*R+1);
fffilt = fft2(ffilt,4*R+1, 4*R+1);

patch = ifft2(sum(bsxfun(@times,permute(reshape(ccfilt,4*R+1,4*R+1,Nc,1,numflt),[1 2 4 3 5]) ...
    , reshape(fffilt,4*R+1,4*R+1,Nc,1,numflt)),5));

if opt == 'S'       % for S matrix
    Nic = fft2(circshift(padarray(patch, [N1-1-2*R N2-1-2*R],'post'),[-4*R-rem(N1,2) -4*R-rem(N2,2)]));
else                % for C matrix
    Nic = fft2(circshift(padarray(patch, [N1-1-2*R N2-1-2*R], 'post'),[-2*R -2*R]));
end
end

