function [nmse] = nmse(x,ref)

nmse = sum(((x(:)-ref(:)).^2))/sum(ref(:).^2);
end