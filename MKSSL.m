%% Xun Yang
function [Method, kernelparam, trainFea]= MKSSL(galX, probX, galLabels, probLabels, option)
kernelFunc= option.kernelFunc;
eta=option.eta;

theta= option.theta;
ratio= option.labelRatio;

k=option.k;
nIter= option.nIter;

ST=ceil(length(unique(galLabels))*ratio)*length(galLabels)/length(unique(galLabels));
d=ceil(length(unique(galLabels))*ratio)-1;

labeltrain=[galLabels(1:ST); probLabels(1:ST)];
Kideal=double(bsxfun(@eq, labeltrain, labeltrain'));
 

 [Ktrain, kernelparam, trainFea] = MultiKernelComputeTrain( galX, probX, ST, Kideal, kernelFunc);

 Y = bsxfun(@eq, galLabels(1:ST), probLabels(1:ST)');
 clear  galX1  probX1 galX2  probX2 galLabels probLabels
 Y = double(Y);

 Wzero= zeros(size(Y));
 W=[Wzero Y; Y Wzero];

 W=W-diag(diag(W));

 D=sum(W);
 D=diag(D);
 L=D-W;
 N=size(Ktrain,1);


  KLK= Ktrain(:,1:2*ST)*L*Ktrain(:,1:2*ST)';
  KDK=Ktrain(:,1:2*ST)*D*Ktrain(:,1:2*ST)';

  [V,~]=eigs(KDK, KLK+theta*trace(KLK)/(N)*eye(N),d);
  
  UnlTrain=Ktrain(2*ST+1:end,:);

  for Iter=1:nIter 
       uKtrain =UnlTrain*V; 
       GalX=  uKtrain(1:end/2,:);
       ProbX=uKtrain(end/2+1:end,:);
       dist   = pdist2(GalX, ProbX, 'cosine');  
       Ws_temp = 1 - (dist);  
       Ws = zeros(size(Ws_temp));
       m=size(GalX,1);
            
       for i = 1:m
             [~, idx]   = sort(Ws_temp(i,:), 'descend');
             Ws(i,idx(1:k)) = Ws_temp(i, idx(1:k));
       end
      Ws = (Ws+Ws')/2; 
      Ws(Ws>0.5) = 1; % Remove noise
      Ws(Ws<0.5 & Ws ~=0)= 0;
     
%    
      if Iter==1
          tempW=Ws;
      else
          Wdist=pdist2(tempW(:)',Ws(:)','cosine');
         % fprintf('Iter=%d, similarity=%f\n', Iter, 1-Wdist);
             fprintf('Iter=%d\n', Iter);
          tempW=Ws;
      end     
   
       ZERO = zeros(m,m);
       sW = [ZERO, Ws; Ws ZERO];  
       sW=sW-diag(diag(sW));
       sD=sum(sW);
       sD=diag(sD);
       sL=sD-sW;

        sKLK=  UnlTrain'*sL*UnlTrain;
        sKDK=  UnlTrain'*sD*UnlTrain;
           Mw=  KLK+eta*sKLK;
         Mb=   KDK+eta*sKDK;%   
         
         [V, ~]= eigs(Mb, Mw+theta*trace(Mw)/N*eye(N),  d); %

        if (Iter>1)&&(1-Wdist)>=0.98;
             break;
        end
  end

    Method.P=V;  
     return;
