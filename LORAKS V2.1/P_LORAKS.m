function [recon] = P_LORAKS(kData, kMask, rank, R, LORAKS_type, lambda, alg, tol, max_iter, VCC)
% This function provides capabilities to solve single-channel and multi-channel
% LORAKS reconstruction problems using one of the formulations from either
% Eq. (5) (which uses LORAKS as regularization and does not require strict data-
% consistency) or Eq. (6) (which enforces strict data-consistency) from the
% technical report:
%
% [1] T. H. Kim, J. P. Haldar.  LORAKS Software Version 2.0:
%     Faster Implementation and Enhanced Capabilities.  University of Southern
%     California, Los Angeles, CA, Technical Report USC-SIPI-443, May 2018.
%
% The problem formulations implemented by this function were originally reported
% in:
%
% [2] J. P. Haldar. Low-Rank Modeling of Local k-Space Neighborhoods (LORAKS)
%     for Constrained MRI. IEEE Transactions on Medical Imaging 33:668-681,
%     2014.
%
% [3] J. P. Haldar, J. Zhuo. P-LORAKS: Low-Rank Modeling of Local k-Space
%     Neighborhoods with Parallel Imaging Data. Magnetic Resonance in Medicine
%     75:1499-1514, 2016.
%
% *********************
%   Input Parameters:
% *********************
%
%    kData: A 3D (size N1 x N2 x Nc) array of measured k-space data to be
%           reconstructed.  The first two dimensions correspond to k-space
%           positions, while the third dimension corresponds to the channel
%           dimension for parallel imaging.  Unsampled data samples should be
%           zero-filled.  The software will use the multi-channel formulation if
%           Nc > 1, and will otherwise use the single-channel formulation.
%
%    kMask: A binary mask of size N1 x N2 that corresponds to the same k-space
%           sampling grid used in kData.   Each entry has value 1 if the
%           corresponding k-space location was sampled and has value 0 if that
%           k-space location was not measured.
%
%    rank:  The matrix rank value used to define the non-convex regularization
%           penalty from Eq. (2) of Ref. [1].
%
%    R: The k-space radius used to construct LORAKS neighborhoods. If not
%       specified, the software will use R=3 by default.
%
%    LORAKS_type: A string that specifies the type of LORAKS matrix that will
%                 be used in reconstruction.  Possible options are: 'C', 'S',
%                 and 'W'.  If not specified, the software will use
%                 LORAKS_type='S' by default.
%
%    lambda: The regularization parameter from Eq. (5).  If lambda=0, the
%            software will use the data-consistency constrained formulation from
%            Eq. (6) instead.  If not specified, the software will use lambda=0
%            by default.
%
%    alg: A parameter that specifies which algorithm the software should use for
%         computation. There are four different options:
%            -alg=1: This choice will use the additive half-quadratic algorithm,
%                    as described in Eq. (18) or (20) of Ref. [1].
%            -alg=2: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (26) or (28) of Ref. [1].
%                    This version does NOT use FFTs.
%            -alg=3: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (26) or (28) of Ref. [1].
%                    This version uses FFTs without approximation, as in
%                    Eq. (35) of Ref. [1].
%            -alg=4: This choice will use the multiplicative half-quadratic
%                    algorithm, as described in Eq. (26) or (28) of Ref. [1].
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
%    VCC: The software will use virtual conjugate coils if VCC=1, and otherwise
%         will not.  If not specified, the software will use VCC=0 by default.
%
% **********************
%   Output Parameters:
% **********************
%
%    recon: The array (size N1 x N2 x Nc) of reconstructed k-space data.
%
% This software is available from
% <a href="matlab:web('http://mr.usc.edu/download/LORAKS2/')">http://mr.usc.edu/download/LORAKS2/</a>.
% As described on that page, use of this software (or its derivatives) in
% your own work requires that you at least cite [1] and either [2] or [3].
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

if not(exist('lambda','var')) || isempty(lambda) || lambda == 0
    lambda_on = false; % enforce exact data consistency
else
    lambda_on = true;
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

if not(exist('VCC','var')) || isempty(VCC)
    VCC = 0;
end

%% Data settings / Virtual conjugate coils augmentation
[N1, N2, Nc] = size(kData);

kMask = repmat(kMask, [1 1 Nc]);

if VCC
    kMask = cat(3, kMask, flip(flip(kMask,1),2));
    kData = cat(3, kData, conj(flip(flip(kData,1),2)));
    Nc = 2*Nc;
end

data = vect(kData.*kMask);

%% P-LORAKS
if Nc>1
    disp('P-LORAKS Reconstruction');
else
    disp('LORAKS Reconstruction');
end

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


if lambda_on
    if alg ~= 4
        AhA = kMask(:) + lambda*mm;
    else
        AhA = kMask(:);
    end
    AhA_dagger = AhA;
    AhA_dagger(find(AhA)) = 1./AhA(find(AhA));
else
    S.type='()';
    S.subs{:} = find(~kMask(:));
    tmp = zeros([N1 N2 Nc],'like',kData);
    A = @(x) data(:) + vect(subsasgn(tmp,S,x));    % missing data embedding operator
    mm_r = subsref(mm,S);
end

if LORAKS_type == 1 && alg == 3
    delta = zeros([N1+2*R N2+2*R],'like',kData);
    delta(1,1) = 1;
    ph = fft2(circshift(delta, [-rem(N1,2) -rem(N2,2)]));
    Ic = @(x) x - conj(x).*repmat(ph,[1 1 1 size(x,4)]);  % multiply shift by -1 -> linear phase
end


%% Reconstruction
ZD = @(x) padarray(reshape(x,[N1 N2 Nc]),[2*R, 2*R], 'post');
ZD_H = @(x) x(1:N1,1:N2,:,:);
MC = @(x) sum(x,3);
MC_H = @(x) repmat(x,[1 1 Nc]);
T = false(N1,N2);
T(R+1+(~rem(N1,2)):N1-R, R+1+(~rem(N2,2)):N2-R) = 1;
T = padarray(T,[R R]);

z = data(:).*kMask(:); % Initialization with zero-filled data

if ~lambda_on
end

for iter = 1:max_iter
    z_cur = z;
    MM = P_M(z);

    if lambda_on % Regularized version
        
        if alg == 1 % Additive
            pmm = svd_left(MM,rank)'; % subspace
            MMr = Ph_M(pmm'*pmm*MM);
            z = AhA_dagger.*(data(:)+lambda*MMr); 
            
        elseif alg == 2 % Multiplicative
            pmm = svd_left(MM,rank)'; % subspace
            M = @(x) AhA.*x-lambda*Ph_M(pmm'*(pmm*P_M(x)));                
            [z,~] = pcg(M, data(:), 1e-6, 50);
            
        elseif alg == 3 % Multiplicative with FFT
            pmm = svd_left(MM,rank)'; % subspace
            nf = size(pmm,1);

            if LORAKS_type == 1 % S                
                pmm = reshape(pmm,[nf, patchSize, 2*Nc]);
                pmm_h = reshape(pmm(:,:,1:2:end)+1j*pmm(:,:,2:2:end),[nf, patchSize*Nc]);
                ffilt = fft2(padfilt(pmm_h,N1,N2,Nc,R));
                M = @(x) AhA.*x-lambda*vect(ZD_H(sum(ifft2(conj(ffilt).*MC_H(Ic(fft2(repmat(T,[1 1 1 nf]).*ifft2(Ic(MC(ffilt.*fft2(repmat(ZD(x),[1 1 1 nf]))))))))),4)));                
            elseif LORAKS_type == 2 % C
                ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
                M = @(x) AhA.*x -lambda*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(x),[1 1 1 nf])))))),4))));                
            elseif LORAKS_type == 3 % W
                ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
                M = @(x) AhA.*x-lambda*(conj(W1(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W1(:).*x),[1 1 1 nf])))))),4))))+conj(W2(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W2(:).*x),[1 1 1 nf])))))),4)))));                
            end
            [z,~] = pcg(M, data(:), 1e-6, 50);
            
        elseif alg == 4 % Multiplicative with FFT (fast approximate)
            Um = svd_left(MM);
            nmm = Um(:,rank+1:end)'; % null space

            if LORAKS_type == 1 % S
                nf = size(nmm,1);
                nmm = reshape(nmm,[nf, patchSize, 2*Nc]);
                nmm_h = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, patchSize*Nc]);
                Nis = filtfilt(nmm_h,'C',N1,N2,Nc,R);
                Nis2 = filtfilt(nmm_h,'S',N1,N2,Nc,R);
                M = @(x) AhA.*x + 2*lambda*(vect(ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(x)),[1 1 1 Nc]),3)))))-vect(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(x))),[1 1 1 Nc]),3))))));                
            elseif LORAKS_type == 2 % C
                Nic = filtfilt(nmm,'C',N1,N2,Nc,R);
                M = @(x) AhA.*x + lambda*vect(ZD_H(ifft2(squeeze(sum(Nic.*repmat(fft2(ZD(x)),[1 1 1 Nc]),3)))));                
            elseif LORAKS_type == 3 % W
                Niw = filtfilt(nmm,'C',N1,N2,Nc,R);
                M = @(x) AhA.*x + lambda*(conj(W1(:)).*vect(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W1(:).*x)),[1 1 1 Nc]),3)))))+ conj(W2(:)).*vect(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W2(:).*x)),[1 1 1 Nc]),3))))));
            end
            [z,~] = pcg(M, data(:), 1e-6, 50);
            
        else
            error('Error: Invalid alg');
        end
        
    else % exact data consistency
        
        if alg == 1 % Additive
            pmm = svd_left(MM,rank)'; % subspace                
            MMr = Ph_M(pmm'*pmm*MM);  

            M_dagger = mm_r;
            M_dagger(find(mm_r)) = 1./mm_r(find(mm_r));
            z = M_dagger.*subsref(MMr,S);                

        elseif alg == 2 % Multiplicative
            pmm = svd_left(MM,rank)'; % subspace
            M = @(x) mm_r.*x - subsref(Ph_M(pmm'*(pmm*P_M(subsasgn(tmp,S,x)))),S);
            b = -mm_r.*subsref(data,S) + subsref(Ph_M(pmm'*(pmm*P_M(data))),S);                
            [z,~] = pcg(M, b, 1e-6, 50);

        elseif alg == 3 % Multiplicative with FFT
            pmm = svd_left(MM,rank)';  % subspace
            nf = size(pmm,1);
            
            if LORAKS_type == 1 % S
                pmm = reshape(pmm,[nf, patchSize, 2*Nc]); % subspace
                pmm_h = reshape(pmm(:,:,1:2:end)+1j*pmm(:,:,2:2:end),[size(pmm,1), size(pmm,2)*Nc]);
                ffilt = fft2(padfilt(pmm_h,N1,N2,Nc,R));
                
                M = @(x) mm_r.*x - subsref(ZD_H(sum(ifft2(conj(ffilt).*MC_H(Ic(fft2(repmat(T,[1 1 1 nf]).*ifft2(Ic(MC(ffilt.*fft2(repmat(ZD(subsasgn(tmp,S,x)),[1 1 1 nf]))))))))),4)),S);
                b = -mm_r.*subsref(data,S) + subsref(ZD_H(sum(ifft2(conj(ffilt).*MC_H(Ic(fft2(repmat(T,[1 1 1 nf]).*ifft2(Ic(MC(ffilt.*fft2(repmat(ZD(data),[1 1 1 nf]))))))))),4)),S);
            elseif LORAKS_type == 2 % C
                ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
                M = @(x) mm_r.*x - subsref(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(subsasgn(tmp,S,x)),[1 1 1 nf])))))),4))),S);
                b = -mm_r.*subsref(data,S) + subsref(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(data),[1 1 1 nf])))))),4))),S);
            elseif LORAKS_type == 3 % W
                ffilt = fft2(padfilt(pmm,N1,N2,Nc,R));
                M = @(x) mm_r.*x - subsref(conj(W1(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W1.*subsasgn(tmp,S,x)),[1 1 1 nf])))))),4)))),S)...
                    -subsref(conj(W2(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W2.*subsasgn(tmp,S,x)),[1 1 1 nf])))))),4)))),S);
                b = -mm_r.*subsref(data,S) + subsref(conj(W1(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W1(:).*data),[1 1 1 nf])))))),4)))),S)...
                    + subsref(conj(W2(:)).*vect(ZD_H(ifft2(sum(conj(ffilt).*MC_H(fft2(repmat(T,[1 1 1 nf]).*ifft2(MC(ffilt.*fft2(repmat(ZD(W2(:).*data),[1 1 1 nf])))))),4)))),S);
            end
            [z,~] = pcg(M, b, 1e-6, 50);
            
        elseif alg == 4 % Multiplicative with FFT (fast approximate)
            Um = svd_left(MM);         % SVD (singular vectors)
            nmm = Um(:,rank+1:end)';

            if LORAKS_type == 1 % S
                nf = size(nmm,1);
                nmm = reshape(nmm,[nf, patchSize, 2*Nc]); % null space
                nmm_h = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, patchSize*Nc]);
                Nis = filtfilt(nmm_h,'C',N1,N2,Nc,R);
                Nis2 = filtfilt(nmm_h,'S',N1,N2,Nc,R);
                M = @(x) 2*subsref(ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(subsasgn(tmp,S,x))),[1 1 1 Nc]),3)))),S) ...
                    -2*subsref(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(subsasgn(tmp,S,x)))),[1 1 1 Nc]),3)))),S);
                b = -2*subsref(ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(data)),[1 1 1 Nc]),3)))),S) ...
                    +2*subsref(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(data))),[1 1 1 Nc]),3)))),S);
            elseif LORAKS_type == 2 % C
                Nic = filtfilt(nmm,'C',N1,N2,Nc,R);
                M = @(x) subsref(ZD_H(ifft2(squeeze(sum(Nic.*repmat(fft2(ZD(subsasgn(tmp,S,x))),[1 1 1 Nc]),3)))),S);
                b = -subsref(ZD_H(ifft2(squeeze(sum(Nic.*repmat(fft2(ZD(data)),[1 1 1 Nc]),3)))),S);
            elseif LORAKS_type == 3 % W
                Niw = filtfilt(nmm,'C',N1,N2,Nc,R);
                M = @(x) subsref(conj(W1).*(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W1.*subsasgn(tmp,S,x))),[1 1 1 Nc]),3)))))...
                    + conj(W2).*(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W2.*subsasgn(tmp,S,x))),[1 1 1 Nc]),3))))),S);
                b = -subsref(conj(W1).*(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W1(:).*data)),[1 1 1 Nc]),3)))))...
                    + conj(W2).*(ZD_H(ifft2(squeeze(sum(Niw.*repmat(fft2(ZD(W2(:).*data)),[1 1 1 Nc]),3))))),S);
            end
            [z,~] = pcg(M, b, 1e-6, 50);
            
        else
            error('Error: Invalid alg');
        end
        
        z = A(z);
        
    end
    
    % check for convergence
    t = (norm(z_cur-z)/norm(z));
    if t < tol
        disp('Convergence tolerance met: change in solution is small');
        break;
    end
    
    % display the status
    if ~rem(iter,1)
        disp(['iter ' num2str(iter) ', relative change in solution: ' num2str(t)]);
    end
    
    if iter==max_iter
        disp('Maximum number of iterations reached');
    end
    
end

recon = reshape(z, [N1 N2 Nc]);

% Remove virtual conjugate coils
if VCC
    Nc = Nc/2;
    recon = recon(:,:,1:Nc);
end
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
function Nic = filtfilt(ncc, opt, N1, N2, Nc, R)
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

filtfilt = zeros([(2*R+1)*(2*R+1),Nc,numflt],'like',ncc);
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
