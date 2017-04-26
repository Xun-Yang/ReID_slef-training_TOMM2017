function [Ktrain, kernelparam, trainFea] = MultiKernelComputeTrain( galX, probX, ST, Kideal,kernelFunc)
% 
%%   
   
       LabeledtrainFea    =   [galX(1:ST,:);  probX(1:ST,:)];
       unLabeldtrainFea   =   [galX(ST+1:end,:); probX(ST+1:end,:)];

       trainFea=[LabeledtrainFea; unLabeldtrainFea];
       trainNum=size(trainFea,1);
       sq_dist = EuclidDist(trainFea, trainFea);
       gamma=2:0.1:3;
       mu=mean(mean(sq_dist));
       for c=1: length(gamma)     
          Ktemp{c}= ComputeKernel(trainFea, kernelFunc{1}, gamma(c), mu,sq_dist);       
       end  
      
        kernelNum=  length(Ktemp);
        Atemp= zeros(kernelNum,1);

         for i=1: kernelNum
                B=Ktemp{i};
            	Klabeled=B(1:2*ST,1:2*ST);
	            Atemp(i)=trace(Klabeled'*Kideal)/(sqrt(trace(Kideal'*Kideal))*sqrt(trace(Klabeled'*Klabeled)));
         end
                K=Ktemp;
                A=Atemp;
      
         weight=zeros(kernelNum,1);
         Ktrain=zeros(trainNum, trainNum);      
         
         for i=1: kernelNum
 	         weight(i)=A(i)/sum(A);   
            Ktrain=Ktrain+weight(i)*K{i};
         end
 
        kernelparam.weight=weight;
        kernelparam.mu=mu;
        kernelparam.gamma=gamma;
end

