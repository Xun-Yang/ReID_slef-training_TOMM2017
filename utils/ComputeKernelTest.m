% Calculate the kernel matrix for train and test set.
% TODO: Replace the ComputeKernel function in  ComputeKernel.m
% Input: 
%       Method: the distance learning algorithm struct. In this function
%               only field used "kernel", the name of the kernel function. 
%       train: The data used to learn the projection matric. Each row is a
%               sample vector. Ntr-by-d
%       test: The data used to test and calculate the CMC for the
%               algorithm. Each row is a sample vector. Nts-by-d
function [K_test] = ComputeKernelTest(train, test, kernel, gamma, mu, sq_dist)

if (size(train,2))>2e4 && (strcmp(kernel, 'chi2')|| strcmp(kernel, 'chi2-rbf'))% )
    % if the input data matrix is too large then use parallel computing
    % tool box.
    
    switch kernel
        case {'linear'}
            K_test = train * test';
        case {'chi2'}
         parpool
            parfor i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
        delete(gcp);    
          clear subp sump;
          K_test=double(K_test);
        case {'Gaussian'}% Gaussian kernel
          sq_dist = EuclidDist(train, train);
          mu=mean(mean(sq_dist));
          clear sq_dist
          
          sq_dist = EuclidDist(train, test);
          K_test = exp(-sq_dist/(2*mu)); 

        case {'chi2-rbf'}
            sigma = 1;
           parpool
            parfor i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);  K_test=double(K_test);
            delete(gcp);   
    end
else
    switch kernel
        case {'linear'}
            K_test =(train * test');
        case {'chi2'}
            for i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
              K_test=double(K_test);
         case {'Gaussian'}% Gaussian kernel

          K_test = exp(-sq_dist/(gamma*mu)); 
        case {'chi2-rbf'}
            sigma = 1;

            for i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);  K_test=double(K_test);
     
    end
 end
return;