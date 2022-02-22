function [x,cost] = softThresh(y,t)
% Complex soft thresholding. If there is only one image, this is equal to 
% regular complex threholding. If there are multiple images size(y)>1, it
% applies joint sparsity soft-thresholding.
absy = sqrt(sum(abs(y).^2,3));
unity = y./(repmat(absy,[1,1,size(y,3)])+eps);

res = absy-t;
res = (res + abs(res))/2;
x = unity.*repmat(res,[1,1,size(y,3)]);
cost = sqrt(sum(abs(x).^2,3));
cost=sum(cost(:));