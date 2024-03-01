%Code for selection of important features
clear
clc
% close all

%Read data into array and variable names into separate vector
T = readtable('PlateResults2.txt',VariableNamingRule='preserve');
names = T.Properties.VariableNames;
Tarray = table2array(T);
reducedTarray = Tarray(:,8:2139);

%Normalization
normArray = normalize(reducedTarray,"range");

arraySize = size(Tarray);

compound_data = [];
compound_labels = [];

sick_data = [];
sick_labels = [];

healthy_data = [];
healthy_labels = [];

for i = 1:arraySize(1)
    if Tarray(i,2) < 5
        %Compound
        compound_data = vertcat(compound_data,normArray(i,:));
        compound_labels = vertcat(compound_labels,2);
    elseif Tarray(i,2) < 8
        %Sick
        sick_data = vertcat(sick_data,normArray(i,:));
        sick_labels = vertcat(sick_labels,0);
    elseif Tarray(i,2) > 9
        %Healthy 
        healthy_data = vertcat(healthy_data,normArray(i,:));
        healthy_labels = vertcat(healthy_labels,1);
    end
end

%Best sick_compound separators
[sick_features, sick_scores] = fscmrmr(vertcat(sick_data,compound_data),vertcat(sick_labels,compound_labels));

%Worst healthy_compound separators
[healthy_features, healthy_scores] = fscmrmr(vertcat(healthy_data,compound_data),vertcat(healthy_labels,compound_labels));
healthy_features = flip(healthy_features);

%Best of both worlds
array_length = length(sick_features);
hyperparameter_tuning_output = [];

for phi = 1:1:20
    best_feature_indices = zeros(1,array_length);
    for i = 1:length(sick_features)
        best_feature_indices(sick_features(i)) = best_feature_indices(sick_features(i)) + i;
        best_feature_indices(healthy_features(i)) = best_feature_indices(healthy_features(i)) + phi*i;
    end
    
    [blank,best_feature_indices] = sort(best_feature_indices);
    assayZ = 0;
    numfeatures = 1;

    while(assayZ < 0.2)
        numfeatures = numfeatures + 1;
    
        %Select top features
        best_sick_data = [];
        best_healthy_data = [];
        best_compound_data = [];
        for i = 1:numfeatures
            best_sick_data = horzcat(best_sick_data,sick_data(:,best_feature_indices(i)));
            best_healthy_data = horzcat(best_healthy_data,healthy_data(:,best_feature_indices(i)));
            best_compound_data = horzcat(best_compound_data,compound_data(:,best_feature_indices(i)));
        end
        
        [LDAdata, W, lambda] = LDA(vertcat(best_sick_data,best_compound_data), vertcat(sick_labels,compound_labels));
        
        labels = vertcat(sick_labels,compound_labels);
        healthy = best_healthy_data*W;
        healthy = healthy(:,1);
        
        arraySize = size(LDAdata);
        sick = [];
        compound = [];
        
        for k = 1:arraySize(1)
            if labels(k) == 0
                sick = vertcat(sick,LDAdata(k,1));
            else
                compound = vertcat(compound,LDAdata(k,1));
            end
        end
        
        %Z'-Factor
        
        sSTD = std(sick);
        sMean = mean(sick);
        hSTD = std(healthy);
        hMean = mean(healthy);
        cSTD = std(compound);
        cMean = mean(compound);
        
        assayZ = 1 - 3 * (sSTD + hSTD) / abs(sMean - hMean);
        compoundZ = 1 - 3 * (cSTD + sSTD) / abs(cMean - sMean);

    end
    hyperparameter_tuning_output = vertcat(hyperparameter_tuning_output,[phi,numfeatures,compoundZ,assayZ]);
end

[B,best_idx] = max(hyperparameter_tuning_output(:,3));
phi = hyperparameter_tuning_output(best_idx,1);
numfeatures = hyperparameter_tuning_output(best_idx,2);

best_feature_indices = zeros(1,array_length);
for i = 1:length(sick_features)
    best_feature_indices(sick_features(i)) = best_feature_indices(sick_features(i)) + i;
    best_feature_indices(healthy_features(i)) = best_feature_indices(healthy_features(i)) + phi*i;
end
    
[blank,best_feature_indices] = sort(best_feature_indices);

%Select top features
best_sick_data = [];
best_healthy_data = [];
best_compound_data = [];
for i = 1:numfeatures
    best_sick_data = horzcat(best_sick_data,sick_data(:,best_feature_indices(i)));
    best_healthy_data = horzcat(best_healthy_data,healthy_data(:,best_feature_indices(i)));
    best_compound_data = horzcat(best_compound_data,compound_data(:,best_feature_indices(i)));
end

tiledlayout(2,2)
%2D LDA
[LDAdata, W, lambda] = LDA(vertcat(best_sick_data,best_healthy_data,best_compound_data), vertcat(sick_labels,healthy_labels,compound_labels));

labels = vertcat(sick_labels,healthy_labels,compound_labels);
LDAdata = -1 .* LDAdata;
arraySize = size(LDAdata);
sick = [];
healthy = [];
compound = [];

for k = 1:arraySize(1)
    if labels(k) == 0
        sick = vertcat(sick,LDAdata(k,1:2));
    elseif labels(k) == 1
        healthy = vertcat(healthy,LDAdata(k,1:2));
    else
        compound = vertcat(compound,LDAdata(k,1:2));
    end
end

nexttile
hold on
scatter(compound(:,1),compound(:,2),'filled')
scatter(sick(:,1),sick(:,2),'filled')
scatter(healthy(:,1),healthy(:,2),'filled')
legend("Compound","Stressed","Healthy")
title('2D LDA')
hold off

%1D LDA
[LDAdata, W, lambda] = LDA(vertcat(best_sick_data,best_compound_data), vertcat(sick_labels,compound_labels));

labels = vertcat(sick_labels,compound_labels);
healthy = best_healthy_data*W;
healthy = healthy(:,1);

arraySize = size(LDAdata);
sick = [];
compound = [];

for k = 1:arraySize(1)
    if labels(k) == 0
        sick = vertcat(sick,LDAdata(k,1));
    else
        compound = vertcat(compound,LDAdata(k,1));
    end
end

%Z'-Factor

sSTD = std(sick);
sMean = mean(sick);
hSTD = std(healthy);
hMean = mean(healthy);
cSTD = std(compound);
cMean = mean(compound);

assayZ = 1 - 3 * (sSTD + hSTD) / abs(sMean - hMean);
compoundZ = 1 - 3 * (cSTD + sSTD) / abs(cMean - sMean);

%Plot LDA Data
nexttile
hold on
scatter(compound,zeros(size(compound)),'filled')
scatter(sick,zeros(size(sick)),'filled')
scatter(healthy,zeros(size(healthy)),'filled')
title("LDA Plot assay Z' = " + assayZ + " compound Z' = " + compoundZ) 
legend("Compound","Stressed","Healthy")
hold off

%Plot Gaussian Distributions
nexttile
hold on
x1 = (-5 * cSTD:0.00001:5 * cSTD) + cMean;  %// Plotting range
y1 = exp(- 0.5 * ((x1 - cMean) / cSTD) .^ 2) / (cSTD * sqrt(2 * pi));
plot(x1, y1,"Color","#0072BD")

x2 = (-5 * sSTD:0.00001:5 * sSTD) + sMean;  %// Plotting range
y2 = exp(- 0.5 * ((x2 - sMean) / sSTD) .^ 2) / (sSTD * sqrt(2 * pi));
plot(x2, y2,"Color","#D95319")

x3 = (-5 * hSTD:0.00001:5 * hSTD) + hMean;  %// Plotting range
y3 = exp(- 0.5 * ((x3 - hMean) / hSTD) .^ 2) / (hSTD * sqrt(2 * pi));
plot(x3, y3,"Color","#EDB120")

title('Gaussian Distributions')
legend('Compound','Stressed','Healthy')
xlabel("LDA Value")
ylabel("Probability")
hold off

%PCA
[coeff,PCAdata,latent,~,explained] = pca(vertcat(best_sick_data,best_healthy_data,best_compound_data));
labels = vertcat(sick_labels,healthy_labels,compound_labels);

arraySize = size(PCAdata);
compound = [];
sick = [];
healthy = [];

for k = 1:arraySize(1)
    if labels(k) == 0
        sick = vertcat(sick,PCAdata(k,:));
    elseif labels(k) == 1
        healthy = vertcat(healthy,PCAdata(k,:));
    else
        compound = vertcat(compound,PCAdata(k,:));
    end
end

nexttile
hold on
scatter(compound(:,1),compound(:,2),'filled')
scatter(sick(:,1),sick(:,2),'filled')
scatter(healthy(:,1),healthy(:,2),'filled')
title("PCA")
legend("Compound","Stressed","Vehicle")
xlabel('PC1')
ylabel('PC2')
hold off