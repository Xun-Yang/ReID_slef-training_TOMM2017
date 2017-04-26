function [K, gamma] = GaussianKernel(sq_dist, mu) 
	
 	ratio=2;
% 	count=length(ratio);
% 	gamma=ones(size(ratio));

%       for i=1:count
	
        gamma     = 1/(ratio*mu);
        K{1} = exp(-sq_dist*gamma);
%       end

end
