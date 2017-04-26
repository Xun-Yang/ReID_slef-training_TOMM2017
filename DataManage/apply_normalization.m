
%% Mean removal + L2 norm normalization
if tot == 1; mean_cell = cell(parFea.featurenum, 1); end

for f = 1:parFea.featurenum
    if parFea.usefeature(f) == 1
        feature = feature_cell{f,1};
        X = feature;
        
        if tot == 1 % training data
            meanX = mean( X, 1);
            mean_cell{f} = meanX;
        end
        if tot == 2 % test data
            meanX = mean_cell{f};
        end
        
        Y = ( X - repmat(meanX, size(X,1), 1));
        for dnum = 1:size(X, 1)
            Y(dnum,:) = Y(dnum, :)./norm(Y(dnum, :), 2);
        end
        feature_cell{f,1} = Y;
    end
end