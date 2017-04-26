% calculate the kernel matrix.
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% Input: 
%       Method: the distance learning algorithm struct. In this function
%               two field are used. 
%               rbf_sigma is written while computing the rbf-chi2 kernel
%               matrix.
%               kernel is the name of the kernel function. 
%       X:  Each row is a sample vector. N-by-d


function [K] = ComputeKernel(X, kernel, gamma, mu, sq_dist)
K= single(zeros(size(X,1)));
if (size(X,2))>2e4 && (strcmp(kernel, 'chi2') || strcmp(kernel, 'chi2-rbf'))%
    switch kernel
        case {'linear'}% linear kernel
            K = (X*X');
        case {'chi2'}% chi2 kernel
            parpool    
            parfor i =1:size(X,1)
                dotp = bsxfun(@times, X(i,:), X);
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
            end
           delete(gcp);
            clear subp sump;
            K=double(K);
         case {'Gaussian'}% Gaussian kernel
          sq_dist = EuclidDist(X, X);
          mu=mean(mean(sq_dist));
          K= exp(-sq_dist/(2*mu)); 

        case {'chi2-rbf'}% chi2 RBF kernel
            parpool
            parfor i =1:size(X,1)
                subp = bsxfun(@minus, X(i,:), X);
                subp = subp.^2;
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(sum(subp./(sump+eps),2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            % rbf-chi2 kernel parameter can be set here. For example, the
            % first quartile of the distance can be used to normalize the
            % distribution of the chi2 distance  
            rbf_sigma = 1; %temp(ceil(length(temp)*0.25));
            K =exp( -K/rbf_sigma);
            clear subp sump;
                K=double(K);
          delete(gcp);
    end
 else
    switch kernel
        case {'linear'}% linear kernel
           K = (X*X');
        case {'chi2'}% chi2 kernel
            for i =1:size(X,1)
                dotp = bsxfun(@times, X(i,:), X);
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
            end
            clear subp sump;
            K=double(K);
          case {'Gaussian'}% Gaussian kernel         
        
          K= exp(-sq_dist/(gamma*mu)); 
        case {'chi2-rbf'}% chi2 RBF kernel
            for i =1:size(X,1)
                subp = bsxfun(@minus, X(i,:), X);
                subp = subp.^2;
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(sum(subp./(sump+1e-10),2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            rbf_sigma = 1;
            K =exp( -K/rbf_sigma);
            clear subp sump;
              K=double(K);
    end
end
return;
