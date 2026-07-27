%% CLASSIFICATION_DEMO
% Compares 5 supervised classification algorithms on the classic Fisher
% Iris dataset (built into MATLAB, no download needed):
%   1) Logistic Regression
%   2) Decision Tree
%   3) Random Forest (bagged trees)
%   4) Support Vector Machine (SVM)
%   5) Naive Bayes
%
% To keep it visual, we use only 2 of the 4 iris features (Petal Length
% and Petal Width) so we can plot 2D decision boundaries side by side.
%
% Requires: Statistics and Machine Learning Toolbox

clear; clc; close all;

%% Load data
load fisheriris                  % loads 'meas' (150x4) and 'species' (150x1 cell)
X = meas(:, 3:4);                % Petal Length, Petal Width (most separable pair)
Y = species;                     % 'setosa','versicolor','virginica'
classNames = unique(Y);

rng(1);                          % reproducibility
cv = cvpartition(Y, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);  Ytrain = Y(training(cv));
Xtest  = X(test(cv), :);      Ytest  = Y(test(cv));

% Grid for decision-boundary plotting
[xx, yy] = meshgrid(linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, 200), ...
                     linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, 200));
gridPts = [xx(:), yy(:)];

figure('Name','Classification Comparison','Position',[100 100 1400 800]);

%% 1) Logistic Regression (multiclass via 'multinomial' fitting)
mdlLogit = fitcecoc(Xtrain, Ytrain, 'Learners', ...
    templateLinear('Learner','logistic'));
predLogit = predict(mdlLogit, gridPts);
accLogit  = mean(strcmp(predict(mdlLogit, Xtest), Ytest));
plotBoundary(1, xx, yy, predLogit, classNames, Xtrain, Ytrain, ...
    sprintf('Logistic Regression (acc=%.2f%%)', 100*accLogit));

%% 2) Decision Tree
mdlTree = fitctree(Xtrain, Ytrain);
predTree = predict(mdlTree, gridPts);
accTree  = mean(strcmp(predict(mdlTree, Xtest), Ytest));
plotBoundary(2, xx, yy, predTree, classNames, Xtrain, Ytrain, ...
    sprintf('Decision Tree (acc=%.1f%%)', 100*accTree));
% Uncomment to show the actual tree structure (great for class discussion):

%% 3) Random Forest (bagged decision trees, i.e. TreeBagger)
nTrees = 50;
mdlForest = TreeBagger(nTrees, Xtrain, Ytrain, 'Method', 'classification');
predForest = predict(mdlForest, gridPts);            % cell array of strings
predForestTest = predict(mdlForest, Xtest);           % cell array of strings
accForest = mean(strcmp(predForestTest, Ytest));
figure(1);
plotBoundary(3, xx, yy, cellstr(predForest), classNames, Xtrain, Ytrain, ...
    sprintf('Random Forest, %d trees (acc=%.1f%%)', nTrees, 100*accForest));

%% 4) Support Vector Machine (SVM, multiclass via ECOC + Gaussian kernel)
mdlSVM = fitcecoc(Xtrain, Ytrain, 'Learners', ...
    templateSVM('KernelFunction', 'gaussian'));
predSVM = predict(mdlSVM, gridPts);
accSVM  = mean(strcmp(predict(mdlSVM, Xtest), Ytest));
plotBoundary(4, xx, yy, predSVM, classNames, Xtrain, Ytrain, ...
    sprintf('SVM, Gaussian kernel (acc=%.1f%%)', 100*accSVM));

%% 5) Naive Bayes
mdlNB = fitcnb(Xtrain, Ytrain);
predNB = predict(mdlNB, gridPts);
accNB  = mean(strcmp(predict(mdlNB, Xtest), Ytest));
plotBoundary(5, xx, yy, predNB, classNames, Xtrain, Ytrain, ...
    sprintf('Naive Bayes (acc=%.1f%%)', 100*accNB));

sgtitle('Five Classifiers, Same Data: Look How Different the Boundaries Are');

%% Plot decision tree
figure(2); view(mdlTree, 'Mode', 'graph');

%% ---- Discussion prompts for class ----
% 1) Which boundary looks most "rigid" (blocky)? Which looks smoothest?
%    (Trees/Forest -> axis-aligned rectangles. SVM -> smooth curves.)
% 2) Naive Bayes assumes features are independent -- clearly false for
%    petal length/width (they're correlated!). Why does it still work
%    reasonably well?
% 3) Random Forest is many Decision Trees voting. Compare boundary
%    "smoothness" between panels 2 and 3.

%% ---- Local helper function ----
function plotBoundary(idx, xx, yy, pred, classNames, Xtrain, Ytrain, titleStr)
    subplot(2,3,idx);
    predCat = categorical(pred, classNames);
    gscatter(xx(:), yy(:), predCat, lines(numel(classNames)), '.', 1, 'off');
    
    hold on;
    gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain, 'rgb',[], 20);
    xlabel('Petal Length (cm)'); ylabel('Petal Width (cm)');
    title(titleStr, 'FontSize', 10);
    legend('off');
    axis tight;
    
end
