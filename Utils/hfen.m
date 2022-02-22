function hfenVal = hfen(img,ref)
%HFEN High frequency error norm for measuring image quality
%
% Based on "MR image reconstruction from highly undersampled k-space data 
% by dictionary learning", Ravishankar, Saiprasad; Bresler, Yoram;
% IEEE Transactions on Medical Imaging, 2011
%
% "The high-frequency error norm (HFEN) is used to quantify the quality of 
% reconstruction of edges and fine features. We employ a rotationally 
% symmetric LoG (Laplacian of Gaussian) filter to capture edges. The filter
% kernel is of size 15 15 pixels, and has a standard deviation of 1.5 
% pixels. HFEN is computed as the L2 norm of the result obtained by LoG
% filtering the difference between the reconstructed and reference images.
% Needless to say, these metrics do not necessarily represent perceptual 
% visual quality, which can only be accurately assessed by human visual 
% observer studies. Nonetheless, large differences in these metrics 
% typically correspond to visually perceptible differences."
%
%
% Written by Efe Ilicak, 19/11/2019

hsize = 15;
sigma = 1.5; 
LoG = fspecial('log',hsize,sigma);

difImg = img-ref;
logD = imfilter(difImg,LoG);
hfenVal = norm(logD(:),2);


