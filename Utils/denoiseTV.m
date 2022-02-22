function [x , J] = denoiseTV(y,lambda,Nit,ctype) 
% [x,J] = denoiseTV(y,lambda,Nit,ctype)
% Total variation filtering (denoising) using
% iterative clipping algorithm.
% INPUT
% y - noisy signal (row vector)
% lambda - regularization parameter
% Nit - number of iterations
% ctype - clipping type
% OUTPUT
% x - result of denoising
% J - objective function
if nargin < 3
    Nit = 5;
end
if nargin < 4
    ctype = 0;
    warning('ctype undetermined!')
end

%sizey=size(y);
J = zeros(Nit,size(y,3)); % objective function
%J2 = zeros(Nit,size(y,3)); % objective function
%N = numel(y);%prod(size(y));
z=zeros(2*size(y,1)*size(y,2),size(y,3));
%z = zeros(2*N,1); % initialize z;
alpha = 4;
T = lambda/2;
%fprintf('ctype = %d\n',ctype)
for k = 1:Nit
    parfor ind2=1:size(y,3)
        x(:,:,ind2) = y(:,:,ind2) - adjD(z(:,ind2),size(y(:,:,ind2))); % y - D' z
        fd = D(x(:,:,ind2)); % Dx
        
        %	J2(k,ind2) = sum(sum(abs(x(:,:,ind2)-y(:,:,ind2)).^2)) + lambda * sum(abs(fd(:)));
        J(k,ind2) = sum(abs(fd(:)));
        z(:,ind2) = z(:,ind2) + 1/alpha * fd; % z + 1/alpha D z
    end
    %J(k) = sum(sum(abs(x(:,:,1)-y(:,:,1)).^2)) + lambda * sum(sum(abs(D(x(:,:,1)))));
    %figure;imshow(abs(x(:,:,1)),[],'InitialMagnification',400);title('fd')
    z = sclip(z,T,ctype).*exp(sqrt(-1)*angle(z));
    %z = max(min(abs(z),T),-T); % clip(z,T)
    %disp(['TV Iteration' num2str(k) ' cost:' num2str(J(k,1))])
    %disp(['TV Problem total cost:' num2str(J2(k,1))])
end

return

function zc = sclip(z,T,ctype)
% clipping function
absz = abs(z);
switch ctype
    case 0
        res = absz>T;
        zc = res.*T + (1-res).*absz;
    case 1
        k = 10;
        zc = T.*power(atan(power(absz/T,k)),(1/k));
    case 2
        unity = absz./(absz.*T+eps);
        pres = (absz-T)>0;
        res = (1-pres).*(absz.^2)/2 + (pres).*(absz-T/2).*T;
        zc = unity.*res;
    case 3
        %absj = sqrt(mean(absz.^2,2));%
        %
        mz=mean(absz,2);
        sz=std(absz,0,2);
        nlevel=sz./mz;
        nidx=nlevel<0.9;
        nidx=repmat(nidx,[1,size(absz,2)]);
        %figure(100);plot(sz./mz);
        %figure;subplot(2,1,1);plot((absz(:,1))');title('raw')
        %subplot(2,1,2);plot((absj(:,1))');title('mean added')
        %T=T+mean(mean(absz))
        res = absz>T;
        %keyboard
        res=res|nidx;
        %figure;imagesc(res);colormap gray
        %res=repmat(res,[1,size(absz,2)]);
        zc = res.*T + (1-res).*absz;
        %figure;imagesc(res); colormap gray
        %keyboard
end
return

function res = D(image)
% This function computes the finite difference transform of the image
Dx = image([2:end,end],:) - image;
Dy = image(:,[2:end,end]) - image;
res = [Dx(:);  Dy(:)];
return

function res = adjD(y,ndims)
% Computes the adjoint finite difference transform
N = prod(ndims);
Dx = reshape(y(1:N),ndims);
Dy = reshape(y(N+1:end),ndims);
res = adjDx(Dx) + adjDy(Dy);
%res = res(:);
return;

function res = adjDy(x)
res = x(:,[1,1:end-1]) - x;
res(:,1) = -x(:,1);
res(:,end) = x(:,end-1);
return;

function res = adjDx(x)
res = x([1,1:end-1],:) - x;
res(1,:) = -x(1,:);
res(end,:) = x(end-1,:);
return;
