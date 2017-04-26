function [Kfeature] = KernelTest(train, test, param, kernel) 

weight= param.weight;
gamma=param.gamma;
mu=param.mu;
     
Kfeature=zeros(size(train,1), size(test,1));
            
          sq_dist = EuclidDist(train, test);
  for c=1: length(gamma)    
      K_test{c} = ComputeKernelTest(train, test, kernel{1}, gamma(c), mu, sq_dist);
  end
      
   for i=1:length(K_test)
		Kfeature=Kfeature+weight(i)*K_test{i};		
   end
    
    Kfeature=Kfeature';
end
