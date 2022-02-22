function simwritehigh(im,fname,dwin)
%% simwrite(im,fname,dwin) 
% Save the image in the scale given in dwin
if nargin<3
    mnorm = 0;
else
	mnorm=1;
end

% then write it out
if mnorm == 1
    %imwrite(imadjust(im,dwin),fname);
    imwrite(mat2gray(im, dwin),fname,'tiff','Resolution',[320,320]);
    
else
    imwrite(im,fname,'tiff','Resolution',[320,320]);
end