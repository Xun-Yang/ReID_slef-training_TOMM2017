%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% config.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('GOG/');
addpath('GOG/mex');
addpath('XQDA/');
addpath('DataManage/');


set_database;

% image size for resize
H0 = 128; W0 = 48;

%%?@configuration of features.  
featuresetting = 2; % 1-- GOG_RGB, 2 -- GOG_Fusion
parFea.featurenum = 4; 
parFea.featureConf = cell( parFea.featurenum, 1);
parFea.usefeature = zeros( parFea.featurenum, 1);

switch featuresetting
    case 1 % GOG_RGB
        parFea.usefeature(1) = 1; % GOG_RGB
        parFea.usefeature(2) = 0; % GOG_Lab
        parFea.usefeature(3) = 0; % GOG_HSV
        parFea.usefeature(4) = 0; % GOG_nRnG
    case 2 % GOG_Fusion
        parFea.usefeature(1) = 1; % GOG_RGB
        parFea.usefeature(2) = 1; % GOG_Lab
        parFea.usefeature(3) = 1; % GOG_HSV
        parFea.usefeature(4) = 1; % GOG_nRnG
    otherwise
        fprintf('Undefined feature setting \n');
end

for f = 1:parFea.featurenum
    parFea.featureConf(f) = {set_default_parameter(f)};
end








